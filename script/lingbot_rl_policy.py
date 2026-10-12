import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from lerobot.policies.lingbot_va.modeling_lingbot_va import LingBotVAPolicy
from lerobot.policies.lingbot_va.utils import data_seq_to_patch

from script.lingbot_eval_config import CAMERAS
from script.lingbot_rl_model import ACTION_DIM, HORIZON, STATE_DIM, USED_DOF, apply_residual, mask_unused_dof


def env_action_count(first_chunk):
    return 12 if first_chunk else 16


def model_to_mlp(actions):
    batch = actions.shape[0]
    return actions.squeeze(-1).permute(0, 2, 3, 1).reshape(batch, HORIZON, ACTION_DIM)


def mlp_to_model(chunk):
    batch = chunk.shape[0]
    return chunk.reshape(batch, 4, 4, ACTION_DIM).permute(0, 3, 1, 2).unsqueeze(-1)


def slice_env_actions(mlp_chunk, first_chunk):
    start = 4 if first_chunk else 0
    return mlp_chunk[:, start:, :USED_DOF]


def pool_critic_state(video_tokens, text_tokens):
    if video_tokens.shape[-1] != STATE_DIM or text_tokens.shape[-1] != STATE_DIM:
        raise ValueError("Critic tokens must be 3072-d pre-proj_out / text features")
    return torch.cat([video_tokens, text_tokens], dim=1).mean(dim=1)


def histogram_entropy(samples, bins=32):
    values = samples.detach().float().cpu().numpy()
    if values.ndim == 3:
        values = values.reshape(values.shape[0], -1)
    entropies = []
    for dim in range(values.shape[1]):
        hist, _ = np.histogram(values[:, dim], bins=bins)
        total = hist.sum()
        if total == 0:
            continue
        probs = hist.astype(np.float64) / total
        probs = probs[probs > 0]
        entropies.append(float(-(probs * np.log(probs)).sum()))
    return float(np.mean(entropies)) if entropies else 0.0


def frozen_prior_kwargs():
    return {
        "text_encoder_device": "cpu",
        "device": "cuda",
        "dtype": "bfloat16",
        "attn_mode": "torch",
        "image_hflip": False,
        "camera_layout": "width_concat",
        "height": 128,
        "width": 128,
        "action_per_frame": 4,
        "frame_chunk_size": 4,
        "attn_window": 30,
        "num_inference_steps": 20,
        "video_exec_step": -1,
        "action_num_inference_steps": 50,
        "guidance_scale": 5.0,
        "action_guidance_scale": 1.0,
        "snr_shift": 5.0,
        "action_snr_shift": 0.05,
        "used_action_channel_ids": list(range(7)),
        "obs_cam_keys": list(CAMERAS),
        "save_predicted_video": False,
    }


def reraise_action_batch_failure(exc):
    if "out of memory" in str(exc).lower():
        raise exc
    if str(exc) == "action candidate batching failed":
        raise exc
    raise RuntimeError("action candidate batching failed") from exc


def expand_conditional_kv(transformer, k, plans=1):
    for block in transformer.blocks:
        cache = block.attn1.attn_caches.get("pos") if block.attn1.attn_caches else None
        if cache is None or "k" not in cache or cache["k"].shape[0] < plans:
            raise RuntimeError("action candidate batching failed")
        for key in ("k", "v"):
            cache[key] = cache[key][:plans].repeat_interleave(k, dim=0)


def expand_kv_rows(transformer, reps):
    for block in transformer.blocks:
        cache = block.attn1.attn_caches.get("pos") if block.attn1.attn_caches else None
        if cache is None or "k" not in cache:
            raise RuntimeError("plan batching failed")
        for key in ("k", "v"):
            cache[key] = cache[key].repeat_interleave(reps, dim=0)


def select_kv_rows(transformer, rows):
    index = list(rows)
    for block in transformer.blocks:
        cache = block.attn1.attn_caches.get("pos") if block.attn1.attn_caches else None
        if cache is None or "k" not in cache:
            raise RuntimeError("plan batching failed")
        for key in ("k", "v"):
            cache[key] = cache[key][index]


def _clone_cache(cache):
    return {key: value.clone() if torch.is_tensor(value) else value for key, value in cache.items()}


CHUNK_STATE = (
    "_first_chunk", "_started", "_exec_step", "_frame_st_id", "_obs_buffer", "_executed_actions", "_init_latent",
    "_cache_initialised", "_last_real_latent",
)


class ResidualLingBotPolicy(LingBotVAPolicy):
    def __init__(self, config):
        super().__init__(config)
        self._text_cache = {}
        self.residual_model = None
        self._last_real_latent = None
        self._critic_cache_batches = {}
        self.eval_candidates = 4
        self.eval_plans = 1

    def _get_t5_prompt_embeds(self, prompt, max_sequence_length):
        key = (tuple([prompt] if isinstance(prompt, str) else prompt), max_sequence_length)
        if key not in self._text_cache:
            self._text_cache[key] = super()._get_t5_prompt_embeds(prompt, max_sequence_length).detach()
        return self._text_cache[key].clone()

    def _encode_frames(self, raw_frames):
        latent = super()._encode_frames(raw_frames)
        self._last_real_latent = latent
        return latent

    def commit_executed(self, mlp_chunk, first_chunk=False):
        chunk = mask_unused_dof(mlp_chunk)
        if first_chunk:
            chunk = chunk.clone()
            chunk[:, :4] = 0
        self._executed_actions = mlp_to_model(chunk).to(device=self.config.device, dtype=self.dtype)

    def observe_env_step(self, batch):
        if (self._prev_j + 1) % self._keyframe_stride == 0:
            self._obs_buffer.append(self._extract_raw_obs(batch))
        self._prev_j = self._exec_step % self.config.action_per_frame
        self._exec_step += 1

    def _ensure_critic_cache(self, batch_size=1):
        name = "critic" if batch_size == 1 else "critic_m"
        if self._critic_cache_batches.get(name) == batch_size:
            return name
        cfg = self.config
        latent_h, latent_w = self._latent_hw
        patch = cfg.patch_size
        latent_token_per_chunk = (cfg.frame_chunk_size * latent_h * latent_w) // (patch[0] * patch[1] * patch[2])
        action_token_per_chunk = cfg.frame_chunk_size * cfg.action_per_frame
        with torch.inference_mode(False):
            self.transformer.create_empty_cache(
                name,
                cfg.attn_window,
                latent_token_per_chunk,
                action_token_per_chunk,
                device=self.config.device,
                dtype=self.dtype,
                batch_size=batch_size,
            )
        self._critic_cache_batches[name] = batch_size
        return name

    @torch.no_grad()
    def extract_critic_state(self, batch):
        self.eval()
        self._ensure_frozen_modules()
        self._maybe_init_prompt(batch)
        latent = self._encode_isolated([self._extract_raw_obs(batch)])
        return self._pool_from_latent(latent)

    @torch.no_grad()
    def extract_critic_state_from_latent(self, latent, batch):
        self.eval()
        self._ensure_frozen_modules()
        self._maybe_init_prompt(batch)
        if latent.ndim != 5:
            raise ValueError("Published latents must be shaped (B, C, T, H, W)")
        return self._pool_from_latent(latent.to(device=self.config.device, dtype=self.dtype))

    @torch.no_grad()
    def _pool_from_latent(self, latent):
        b = latent.shape[0]
        name = self._ensure_critic_cache(b)
        captured = []

        def hook(module, inputs, output):
            captured.append(inputs[0])

        handle = self.transformer.proj_out.register_forward_hook(hook)
        try:
            payload = self._prepare_latent_input(latent, None, 0, 0, None, None, frame_st_id=0)["latent_res_lst"]
            payload["text_emb"] = self._prompt_embeds.to(self.dtype).expand(b, -1, -1).clone()
            payload["grid_id"] = payload["grid_id"][None].repeat(b, 1, 1)
            payload["timesteps"] = payload["timesteps"][None].repeat(b, 1)
            self.transformer(payload, update_cache=0, cache_name=name, action_mode=False)
            if not captured:
                raise RuntimeError("Failed to capture pre-proj_out video tokens")
            video_tokens = captured[0].float()
            text_tokens = self.transformer.condition_embedder.text_embedder(
                self._prompt_embeds.to(self.dtype)
            ).float()
            return pool_critic_state(video_tokens, text_tokens.expand(b, -1, -1))
        finally:
            handle.remove()

    def _snapshot_kv(self):
        snaps = []
        for block in self.transformer.blocks:
            cache = block.attn1.attn_caches.get("pos") if block.attn1.attn_caches else None
            if cache is None:
                raise RuntimeError("action candidate batching failed")
            snaps.append(_clone_cache(cache))
        return snaps

    def _restore_kv(self, snaps):
        for block, snap in zip(self.transformer.blocks, snaps):
            block.attn1.attn_caches["pos"] = _clone_cache(snap)

    def _start_raw_obs(self, batch):
        if self._first_chunk:
            if batch is None:
                raise RuntimeError("First chunk requires a live observation batch")
            return self._extract_raw_obs(batch)
        if batch is not None:
            return self._extract_raw_obs(batch)
        if not self._obs_buffer:
            raise RuntimeError("Later chunk is missing keyframe observations")
        return self._obs_buffer[-1]

    def _snapshot_vae_cache(self):
        snaps = {}
        frozen = self._frozen or {}
        for key in ("streaming_vae", "streaming_vae_half"):
            vae = frozen.get(key)
            if vae is None or not hasattr(vae, "feat_cache"):
                continue
            snaps[key] = [item.clone() if torch.is_tensor(item) else item for item in vae.feat_cache]
        return snaps

    def _restore_vae_cache(self, snaps):
        frozen = self._frozen or {}
        for key, cache in snaps.items():
            vae = frozen.get(key)
            if vae is None:
                continue
            vae.feat_cache = [item.clone() if torch.is_tensor(item) else item for item in cache]

    def _clear_vae_cache(self):
        frozen = self._frozen or {}
        for key in ("streaming_vae", "streaming_vae_half"):
            vae = frozen.get(key)
            if vae is not None and hasattr(vae, "clear_cache"):
                vae.clear_cache()

    def _encode_isolated(self, raw_frames):
        snap = self._snapshot_vae_cache()
        try:
            self._clear_vae_cache()
            return self._encode_frames(raw_frames)
        finally:
            self._restore_vae_cache(snap)

    def _snapshot_chunk_state(self):
        kv = []
        for block in self.transformer.blocks:
            cache = block.attn1.attn_caches.get("pos") if block.attn1.attn_caches else None
            kv.append(None if cache is None else _clone_cache(cache))
        return {"kv": kv, "vae": self._snapshot_vae_cache(),
                "attrs": {name: getattr(self, name, None) for name in CHUNK_STATE}}

    def _restore_chunk_state(self, saved):
        for block, cache in zip(self.transformer.blocks, saved["kv"]):
            if block.attn1.attn_caches is not None:
                block.attn1.attn_caches["pos"] = None if cache is None else _clone_cache(cache)
        self._restore_vae_cache(saved["vae"])
        for name, value in saved["attrs"].items():
            setattr(self, name, value)

    def _begin_chunk(self, batch):
        first = self._first_chunk
        start_obs = self._start_raw_obs(batch)
        if first:
            init_latent = self._encode_frames([start_obs])
            self._init_latent = init_latent
            self._init_streaming_cache(init_latent)
            self._obs_buffer = []
            return first, init_latent, 0, init_latent
        critic_latent = self._encode_isolated([start_obs])
        self._compute_kv_cache(self._obs_buffer, self._executed_actions)
        self._obs_buffer = []
        return first, None, self._frame_st_id, critic_latent

    @torch.no_grad()
    def decode_candidates(self, batch, k=4, video_noise=None, action_noise=None):
        self.eval()
        self._ensure_frozen_modules()
        self._maybe_init_prompt(batch)
        first, init_latent, frame_st_id, critic_latent = self._begin_chunk(batch)
        state = self._pool_from_latent(critic_latent)
        actions, latents, z_model, video = self._infer(
            init_latent, frame_st_id, video_noise=video_noise, action_noise=action_noise, k=k
        )
        if first:
            self._first_chunk = False
        self._exec_step = 0
        self._started = True
        return {
            "s": state.float(),
            "z": model_to_mlp(z_model).float(),
            "a_base": model_to_mlp(actions).float(),
            "video_noise": video,
            "latents": latents,
            "first_chunk": first,
        }

    def _video_timesteps(self, exec_step=None):
        cfg = self.config
        self._scheduler.set_timesteps(cfg.num_inference_steps)
        timesteps = F.pad(self._scheduler.timesteps, (0, 1), mode="constant", value=0)
        exec_step = cfg.video_exec_step if exec_step is None else int(exec_step)
        if exec_step != -1:
            if not 1 <= exec_step <= cfg.num_inference_steps:
                raise ValueError("exec_step must lie in [1, num_inference_steps] or be -1")
            timesteps = timesteps[:exec_step]
        return timesteps, exec_step

    def _video_pass(self, latents, t, latent_cond, frame_st_id, write, advance, capture=None):
        cfg = self.config
        latent_h, latent_w = self._latent_hw
        input_dict = self._prepare_latent_input(latents, None, t, t, latent_cond, None, frame_st_id=frame_st_id)
        handle = None
        if capture is not None:
            handle = self.transformer.proj_out.register_forward_hook(
                lambda module, inputs, output: capture.append(inputs[0]))
        try:
            video_noise_pred = self.transformer(
                self._repeat_input_for_cfg(input_dict["latent_res_lst"]),
                update_cache=1 if write else 0,
                cache_name="pos",
                action_mode=False,
            )
        finally:
            if handle is not None:
                handle.remove()
        if advance:
            video_noise_pred = data_seq_to_patch(
                cfg.patch_size,
                video_noise_pred,
                cfg.frame_chunk_size,
                latent_h,
                latent_w,
                batch_size=2 if self._use_cfg else 1,
            )
            if cfg.guidance_scale > 1:
                video_noise_pred = video_noise_pred[1:] + cfg.guidance_scale * (
                    video_noise_pred[:1] - video_noise_pred[1:]
                )
            else:
                video_noise_pred = video_noise_pred[:1]
            latents = self._scheduler.step(video_noise_pred, t, latents, return_dict=False)
        if frame_st_id == 0 and latent_cond is not None:
            latents[:, :, 0:1] = latent_cond
        return latents

    def _video_pass_batched(self, latents, t, latent_cond, frame_st_id, write, advance):
        cfg = self.config
        latent_h, latent_w = self._latent_hw
        m = latents.shape[0]
        cond = None if latent_cond is None else latent_cond.expand(m, -1, -1, -1, -1)
        payload = self._prepare_latent_input(latents, None, t, t, cond, None, frame_st_id=frame_st_id)
        payload = payload["latent_res_lst"]
        payload["noisy_latents"] = payload["noisy_latents"].repeat(2, 1, 1, 1, 1)
        payload["text_emb"] = torch.cat([
            self._prompt_embeds.to(self.dtype).expand(m, -1, -1),
            self._negative_prompt_embeds.to(self.dtype).expand(m, -1, -1)]).clone()
        payload["grid_id"] = payload["grid_id"][None].repeat(2 * m, 1, 1)
        payload["timesteps"] = payload["timesteps"][None].repeat(2 * m, 1)
        pred = self.transformer(payload, update_cache=1 if write else 0, cache_name="pos", action_mode=False)
        if advance:
            pred = data_seq_to_patch(cfg.patch_size, pred, cfg.frame_chunk_size, latent_h, latent_w, batch_size=2 * m)
            pred = pred[m:] + cfg.guidance_scale * (pred[:m] - pred[m:]) if cfg.guidance_scale > 1 else pred[:m]
            latents = self._scheduler.step(pred, t, latents, return_dict=False)
        if frame_st_id == 0 and latent_cond is not None:
            latents[:, :, 0:1] = latent_cond
        return latents

    @torch.no_grad()
    def _infer(self, init_latent, frame_st_id=0, video_noise=None, action_noise=None, k=1):
        cfg = self.config
        device = self.config.device
        latent_h, latent_w = self._latent_hw
        frame_chunk_size = cfg.frame_chunk_size
        latents = (
            video_noise if video_noise is not None else torch.randn(
                1, 48, frame_chunk_size, latent_h, latent_w, device=device, dtype=self.dtype
            )
        ).clone()
        if action_noise is None:
            actions = torch.randn(
                k, cfg.action_dim, frame_chunk_size, cfg.action_per_frame, 1, device=device, dtype=self.dtype
            )
        else:
            actions = action_noise.to(device=device, dtype=self.dtype).clone()
            k = actions.shape[0]
        video_used = latents.clone()
        z_used = actions.clone()

        timesteps, exec_step = self._video_timesteps()
        self._action_scheduler.set_timesteps(cfg.action_num_inference_steps)
        latent_cond = (
            init_latent[:, :, 0:1].to(self.dtype)
            if frame_st_id == 0 and init_latent is not None
            else None
        )
        for i, t in enumerate(timesteps):
            last_step = i == len(timesteps) - 1
            latents = self._video_pass(
                latents, t, latent_cond, frame_st_id, write=last_step, advance=not last_step or exec_step != -1)
        actions = self._decode_actions(actions, frame_st_id, k)
        return actions, latents, z_used, video_used

    def _decode_actions(self, actions, frame_st_id, k, plans=1):
        cfg = self.config
        device = self.config.device
        frame_chunk_size = cfg.frame_chunk_size
        action_timesteps = F.pad(self._action_scheduler.timesteps, (0, 1), mode="constant", value=0)
        if k == 1 and plans == 1:
            for i, t in enumerate(action_timesteps):
                last_step = i == len(action_timesteps) - 1
                action_cond = (
                    torch.zeros(
                        [1, cfg.action_dim, 1, cfg.action_per_frame, 1], device=device, dtype=self.dtype
                    )
                    if frame_st_id == 0
                    else None
                )
                input_dict = self._prepare_latent_input(
                    None, actions[:1], t, t, None, action_cond, frame_st_id=frame_st_id
                )
                action_noise_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict["action_res_lst"]),
                    update_cache=1 if last_step else 0,
                    cache_name="pos",
                    action_mode=True,
                )
                if not last_step:
                    action_noise_pred = rearrange(
                        action_noise_pred, "b (f n) c -> b c f n 1", f=frame_chunk_size
                    )
                    if cfg.action_guidance_scale > 1:
                        action_noise_pred = action_noise_pred[1:] + cfg.action_guidance_scale * (
                            action_noise_pred[:1] - action_noise_pred[1:]
                        )
                    else:
                        action_noise_pred = action_noise_pred[:1]
                    actions = self._action_scheduler.step(
                        action_noise_pred, t, actions[:1], return_dict=False
                    )
                if frame_st_id == 0 and action_cond is not None:
                    actions[:, :, 0:1] = action_cond
            actions[:, ~self._action_mask] *= 0
            return actions

        rows = plans * k
        post_video = self._snapshot_kv()
        try:
            expand_conditional_kv(self.transformer, k, plans)
            for i, t in enumerate(action_timesteps):
                last_step = i == len(action_timesteps) - 1
                action_cond = (
                    torch.zeros(
                        [rows, cfg.action_dim, 1, cfg.action_per_frame, 1], device=device, dtype=self.dtype
                    )
                    if frame_st_id == 0
                    else None
                )
                input_dict = self._prepare_latent_input(
                    None, actions, t, t, None, action_cond, frame_st_id=frame_st_id
                )
                payload = input_dict["action_res_lst"]
                payload["text_emb"] = self._prompt_embeds.to(self.dtype).expand(rows, -1, -1).clone()
                payload["grid_id"] = payload["grid_id"][None].repeat(rows, 1, 1)
                payload["timesteps"] = payload["timesteps"][None].repeat(rows, 1)
                action_noise_pred = self.transformer(
                    payload, update_cache=1 if last_step else 0, cache_name="pos", action_mode=True
                )
                if not last_step:
                    action_noise_pred = rearrange(
                        action_noise_pred, "b (f n) c -> b c f n 1", f=frame_chunk_size
                    )
                    actions = self._action_scheduler.step(action_noise_pred, t, actions, return_dict=False)
                if frame_st_id == 0 and action_cond is not None:
                    actions[:, :, 0:1] = action_cond
            actions[:, ~self._action_mask] *= 0
        except RuntimeError as exc:
            self._restore_kv(post_video)
            reraise_action_batch_failure(exc)
        self._restore_kv(post_video)
        return actions

    @torch.no_grad()
    def decode_plans(self, batch, plans, exec_step, k):
        self.eval()
        self._ensure_frozen_modules()
        self._maybe_init_prompt(batch)
        cfg = self.config
        device = cfg.device
        timesteps, exec_step = self._video_timesteps(exec_step)
        self._action_scheduler.set_timesteps(cfg.action_num_inference_steps)
        saved = self._snapshot_chunk_state()
        try:
            first, init_latent, frame_st_id, critic_latent = self._begin_chunk(batch)
            state = self._pool_from_latent(critic_latent)
            latent_h, latent_w = self._latent_hw
            video_noise = torch.randn(
                plans, 48, cfg.frame_chunk_size, latent_h, latent_w, device=device, dtype=self.dtype)
            action_noise = torch.randn(
                k, cfg.action_dim, cfg.frame_chunk_size, cfg.action_per_frame, 1, device=device, dtype=self.dtype)
            latent_cond = (
                init_latent[:, :, 0:1].to(self.dtype)
                if frame_st_id == 0 and init_latent is not None
                else None
            )
            text_tokens = self.transformer.condition_embedder.text_embedder(self._prompt_embeds.to(self.dtype)).float()
            history = self._snapshot_kv()
            a_base, pools, latents = [], [], []
            for m in range(plans):
                x = video_noise[m:m + 1].clone()
                captured = []
                for i, t in enumerate(timesteps):
                    last = i == len(timesteps) - 1
                    x = self._video_pass(
                        x, t, latent_cond, frame_st_id, write=last, advance=not last or exec_step != -1,
                        capture=captured if last else None)
                a_base.append(model_to_mlp(self._decode_actions(action_noise.clone(), frame_st_id, k)).float())
                self._restore_kv(history)
                pools.append(pool_critic_state(captured[0][:1].float(), text_tokens))
                latents.append(x)
            return {
                "s": state.float(),
                "z": model_to_mlp(action_noise).float(),
                "a_base": torch.stack(a_base),
                "last_pass_pools": torch.cat(pools),
                "video_noise": video_noise,
                "latents": torch.cat(latents),
                "first_chunk": first,
                "exec_step": exec_step,
            }
        finally:
            self._restore_chunk_state(saved)

    @torch.no_grad()
    def select_plan(self, batch, plans, k, rank_fn=None, random_plan=False, decode_all=False):
        if plans == 1:
            decoded = self.decode_candidates(batch, k=k)
            decoded["plan_pools"] = torch.zeros(1, STATE_DIM, device=decoded["s"].device)
            decoded["plan_pool"] = decoded["plan_pools"]
            decoded["plan_index"] = 0
            if decode_all:
                decoded["plan_a_base_all"] = decoded["a_base"][None]
            return decoded
        self.eval()
        self._ensure_frozen_modules()
        self._maybe_init_prompt(batch)
        cfg = self.config
        device = cfg.device
        timesteps, exec_step = self._video_timesteps()
        self._action_scheduler.set_timesteps(cfg.action_num_inference_steps)
        first, init_latent, frame_st_id, critic_latent = self._begin_chunk(batch)
        state = self._pool_from_latent(critic_latent)
        latent_h, latent_w = self._latent_hw
        video_noise = torch.randn(plans, 48, cfg.frame_chunk_size, latent_h, latent_w, device=device, dtype=self.dtype)
        action_noise = torch.randn(
            k, cfg.action_dim, cfg.frame_chunk_size, cfg.action_per_frame, 1, device=device, dtype=self.dtype)
        latent_cond = init_latent[:, :, 0:1].to(self.dtype) if frame_st_id == 0 and init_latent is not None else None
        expand_kv_rows(self.transformer, plans)
        latents = video_noise.clone()
        for i, t in enumerate(timesteps):
            last = i == len(timesteps) - 1
            latents = self._video_pass_batched(
                latents, t, latent_cond, frame_st_id, write=last, advance=not last or exec_step != -1)
        pools = self._pool_from_latent(latents).float()
        if random_plan:
            index = int(np.random.randint(plans))
        else:
            index = 0 if rank_fn is None else int(rank_fn(state.float(), pools))
        if not 0 <= index < plans:
            raise ValueError("rank_fn must return a plan index")
        out = {"s": state.float(), "z": model_to_mlp(action_noise).float(), "plan_pools": pools,
               "plan_pool": pools[index:index + 1], "plan_index": index, "video_noise": video_noise,
               "latents": latents[index:index + 1], "first_chunk": first}
        if decode_all:
            decoded = self._decode_actions(action_noise.repeat(plans, 1, 1, 1, 1), frame_st_id, k, plans)
            decoded = model_to_mlp(decoded).float()
            out["plan_a_base_all"] = decoded.reshape(plans, k, HORIZON, ACTION_DIM)
            out["a_base"] = out["plan_a_base_all"][index]
            select_kv_rows(self.transformer, [index, plans + index])
        else:
            select_kv_rows(self.transformer, [index, plans + index])
            out["a_base"] = model_to_mlp(self._decode_actions(action_noise.clone(), frame_st_id, k)).float()
        if first:
            self._first_chunk = False
        self._exec_step = 0
        self._started = True
        return out

    def _apply_residual_choice(self, decoded, index):
        a_base = decoded["a_base"][index:index + 1].float()
        noise = decoded["z"][index:index + 1].float()
        state = decoded["s"].float()
        if state.shape[0] != 1:
            state = state[:1]
        if self.residual_model is None:
            chosen = a_base
        else:
            chosen = apply_residual(a_base, self.residual_model.actor(state, noise, a_base))
        self.commit_executed(chosen, first_chunk=decoded["first_chunk"])
        if self.config.save_predicted_video:
            self.last_predicted_frames = None
            self.last_predicted_latents = decoded["latents"].detach().to("cpu")
        return slice_env_actions(chosen, decoded["first_chunk"]).to(torch.float32)

    @torch.no_grad()
    def predict_action_chunk(self, batch, **kwargs):
        k = self.eval_candidates if self.residual_model is not None else 1
        plans = getattr(self, "eval_plans", 1) if self.residual_model is not None else 1
        if plans > 1:
            decoded = self.select_plan(batch, plans, k, rank_fn=self.residual_model.rank_plans)
        else:
            decoded = self.decode_candidates(batch, k=k)
        index = 0
        if self.residual_model is not None and decoded["a_base"].shape[0] > 1:
            a_base = decoded["a_base"].float()
            noise = decoded["z"].float()
            state = decoded["s"].float()
            if state.shape[0] != a_base.shape[0]:
                state = state[:1].expand(a_base.shape[0], -1)
            executed = apply_residual(a_base, self.residual_model.actor(state, noise, a_base))
            if plans > 1:
                state = self.residual_model.critic_state(state, decoded["plan_pool"].expand(k, -1))
            index = int(self.residual_model.critic(state, executed).reshape(-1).argmax())
        return self._apply_residual_choice(decoded, index)


def load_residual_policy(checkpoint, model_path, architecture, video_exec_step=-1, video_steps=20, action_steps=50):
    from pathlib import Path

    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.lingbot_va.configuration_lingbot_va import LingBotVAConfig
    from safetensors.torch import load_file

    from script.lingbot_eval import REQUIRED_FILES
    from script.lingbot_eval_config import ARCHITECTURE_KEYS, CAMERAS, validate_sampler_steps

    validate_sampler_steps(video_steps, action_steps, video_exec_step)
    config = LingBotVAConfig(
        **{key: architecture[key] for key in ARCHITECTURE_KEYS},
        input_features={key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)) for key in CAMERAS},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        wan_pretrained_path=str(model_path),
        **{**frozen_prior_kwargs(), "num_inference_steps": video_steps, "video_exec_step": video_exec_step,
           "action_num_inference_steps": action_steps},
    )
    if (
        config.num_inference_steps != video_steps
        or config.video_exec_step != video_exec_step
        or config.action_num_inference_steps != action_steps
        or config.guidance_scale != 5.0
        or config.action_guidance_scale != 1.0
        or config.snr_shift != 5.0
        or config.action_snr_shift != 0.05
        or config.attn_mode != "torch"
        or config.dtype != "bfloat16"
        or config.image_hflip
        or list(config.used_action_channel_ids) != list(range(7))
    ):
        raise ValueError("Residual policy sampler drifted from the 69% SFT eval")
    policy = ResidualLingBotPolicy(config)
    state = load_file(str(Path(checkpoint) / REQUIRED_FILES[0]), device="cpu")
    policy.transformer.load_state_dict(state, strict=True, assign=True)
    del state
    return policy.to("cuda").eval().requires_grad_(False)
