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
    """Sampler/camera kwargs copied from `script.lingbot_eval.load_policy` — the 69% eval."""
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
    """Keep OOM visible. Spec: fail explicitly rather than dropping K or rewriting CUDA errors."""
    if "out of memory" in str(exc).lower():
        raise exc
    if str(exc) == "action candidate batching failed":
        raise exc
    raise RuntimeError("action candidate batching failed") from exc


def expand_conditional_kv(transformer, k):
    """Repeat video-CFG batch index 0 to K. Never expand the uncond row (that would be batch 8)."""
    for block in transformer.blocks:
        cache = block.attn1.attn_caches.get("pos") if block.attn1.attn_caches else None
        if cache is None or "k" not in cache or cache["k"].shape[0] < 1:
            raise RuntimeError("action candidate batching failed")
        for key in ("k", "v"):
            cond = cache[key][:1]
            cache[key] = cond.repeat(k, *([1] * (cond.ndim - 1)))


def _clone_cache(cache):
    return {key: value.clone() if torch.is_tensor(value) else value for key, value in cache.items()}


class ResidualLingBotPolicy(LingBotVAPolicy):
    def __init__(self, config):
        super().__init__(config)
        self._text_cache = {}
        self.residual_model = None
        self._last_real_latent = None
        self._critic_cache_ready = False
        self.eval_candidates = 4

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

    def _ensure_critic_cache(self):
        if self._critic_cache_ready:
            return
        cfg = self.config
        latent_h, latent_w = self._latent_hw
        patch = cfg.patch_size
        latent_token_per_chunk = (cfg.frame_chunk_size * latent_h * latent_w) // (patch[0] * patch[1] * patch[2])
        action_token_per_chunk = cfg.frame_chunk_size * cfg.action_per_frame
        self.transformer.create_empty_cache(
            "critic",
            cfg.attn_window,
            latent_token_per_chunk,
            action_token_per_chunk,
            device=self.config.device,
            dtype=self.dtype,
            batch_size=1,
        )
        self._critic_cache_ready = True

    @torch.no_grad()
    def extract_critic_state(self, batch):
        """Independent real-obs pool. Call after `reset()` (experts). Live AR uses `decode_candidates`['s']."""
        self.eval()
        self._ensure_frozen_modules()
        self._maybe_init_prompt(batch)
        latent = self._encode_isolated([self._extract_raw_obs(batch)])
        return self._pool_from_latent(latent)

    @torch.no_grad()
    def extract_critic_state_from_latent(self, latent, batch):
        """Pool published start-of-chunk VAE latents. Same critic cache as RGB `extract_critic_state`."""
        self.eval()
        self._ensure_frozen_modules()
        self._maybe_init_prompt(batch)
        if latent.ndim != 5:
            raise ValueError("Published latents must be shaped (B, C, T, H, W)")
        return self._pool_from_latent(latent.to(device=self.config.device, dtype=self.dtype))

    @torch.no_grad()
    def _pool_from_latent(self, latent):
        self._ensure_critic_cache()
        captured = []

        def hook(module, inputs, output):
            captured.append(inputs[0])

        handle = self.transformer.proj_out.register_forward_hook(hook)
        try:
            payload = self._prepare_latent_input(latent, None, 0, 0, None, None, frame_st_id=0)["latent_res_lst"]
            payload["grid_id"] = payload["grid_id"][None]
            payload["timesteps"] = payload["timesteps"][None]
            self.transformer(payload, update_cache=0, cache_name="critic", action_mode=False)
            if not captured:
                raise RuntimeError("Failed to capture pre-proj_out video tokens")
            video_tokens = captured[0].float()
            text_tokens = self.transformer.condition_embedder.text_embedder(
                self._prompt_embeds.to(self.dtype)
            ).float()
            return pool_critic_state(video_tokens, text_tokens)
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
        """Live batch on collection; `select_action` later chunks pass `None` and use the last keyframe."""
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
        """1-frame critic encode must not continue the AR streaming-VAE cache (kernel T=3)."""
        snap = self._snapshot_vae_cache()
        try:
            self._clear_vae_cache()
            return self._encode_frames(raw_frames)
        finally:
            self._restore_vae_cache(snap)

    @torch.no_grad()
    def decode_candidates(self, batch, k=4, video_noise=None, action_noise=None):
        self.eval()
        self._ensure_frozen_modules()
        self._maybe_init_prompt(batch)
        first = self._first_chunk
        start_obs = self._start_raw_obs(batch)
        if first:
            init_latent = self._encode_frames([start_obs])
            self._init_latent = init_latent
            self._init_streaming_cache(init_latent)
            self._obs_buffer = []
            frame_st_id = 0
            critic_latent = init_latent
        else:
            critic_latent = self._encode_isolated([start_obs])
            self._compute_kv_cache(self._obs_buffer, self._executed_actions)
            self._obs_buffer = []
            init_latent = None
            frame_st_id = self._frame_st_id
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

        self._scheduler.set_timesteps(cfg.num_inference_steps)
        self._action_scheduler.set_timesteps(cfg.action_num_inference_steps)
        timesteps = F.pad(self._scheduler.timesteps, (0, 1), mode="constant", value=0)
        if cfg.video_exec_step != -1:
            timesteps = timesteps[: cfg.video_exec_step]
        action_timesteps = F.pad(self._action_scheduler.timesteps, (0, 1), mode="constant", value=0)

        for i, t in enumerate(timesteps):
            last_step = i == len(timesteps) - 1
            latent_cond = (
                init_latent[:, :, 0:1].to(self.dtype)
                if frame_st_id == 0 and init_latent is not None
                else None
            )
            input_dict = self._prepare_latent_input(
                latents, None, t, t, latent_cond, None, frame_st_id=frame_st_id
            )
            video_noise_pred = self.transformer(
                self._repeat_input_for_cfg(input_dict["latent_res_lst"]),
                update_cache=1 if last_step else 0,
                cache_name="pos",
                action_mode=False,
            )
            if not last_step or cfg.video_exec_step != -1:
                video_noise_pred = data_seq_to_patch(
                    cfg.patch_size,
                    video_noise_pred,
                    frame_chunk_size,
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

        if k == 1:
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
            return actions, latents, z_used[:1], video_used

        post_video = self._snapshot_kv()
        try:
            expand_conditional_kv(self.transformer, k)
            for i, t in enumerate(action_timesteps):
                last_step = i == len(action_timesteps) - 1
                action_cond = (
                    torch.zeros(
                        [k, cfg.action_dim, 1, cfg.action_per_frame, 1], device=device, dtype=self.dtype
                    )
                    if frame_st_id == 0
                    else None
                )
                input_dict = self._prepare_latent_input(
                    None, actions, t, t, None, action_cond, frame_st_id=frame_st_id
                )
                payload = input_dict["action_res_lst"]
                payload["text_emb"] = self._prompt_embeds.to(self.dtype).expand(k, -1, -1).clone()
                payload["grid_id"] = payload["grid_id"][None].repeat(k, 1, 1)
                payload["timesteps"] = payload["timesteps"][None].repeat(k, 1)
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
        return actions, latents, z_used, video_used

    def _apply_residual_choice(self, decoded, index):
        a_base = decoded["a_base"][index:index + 1].float()
        noise = decoded["z"][index:index + 1].float()
        state = decoded["s"].float()
        if state.shape[0] != 1:
            state = state[:1]
        if self.residual_model is None:
            chosen = a_base
        else:
            chosen = apply_residual(a_base, self.residual_model.actor(state, noise))
        self.commit_executed(chosen, first_chunk=decoded["first_chunk"])
        if self.config.save_predicted_video:
            self.last_predicted_frames = None
            self.last_predicted_latents = decoded["latents"].detach().to("cpu")
        return slice_env_actions(chosen, decoded["first_chunk"]).to(torch.float32)

    @torch.no_grad()
    def predict_action_chunk(self, batch, **kwargs):
        k = self.eval_candidates if self.residual_model is not None else 1
        decoded = self.decode_candidates(batch, k=k)
        index = 0
        if self.residual_model is not None and decoded["a_base"].shape[0] > 1:
            a_base = decoded["a_base"].float()
            noise = decoded["z"].float()
            state = decoded["s"].float()
            if state.shape[0] != a_base.shape[0]:
                state = state[:1].expand(a_base.shape[0], -1)
            executed = apply_residual(a_base, self.residual_model.actor(state, noise))
            index = int(self.residual_model.critic(state, executed).reshape(-1).argmax())
        return self._apply_residual_choice(decoded, index)


def load_residual_policy(checkpoint, model_path, architecture):
    from pathlib import Path

    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.lingbot_va.configuration_lingbot_va import LingBotVAConfig
    from safetensors.torch import load_file

    from script.lingbot_eval import REQUIRED_FILES
    from script.lingbot_eval_config import ARCHITECTURE_KEYS, CAMERAS

    config = LingBotVAConfig(
        **{key: architecture[key] for key in ARCHITECTURE_KEYS},
        input_features={key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)) for key in CAMERAS},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        wan_pretrained_path=str(model_path),
        **frozen_prior_kwargs(),
    )
    if (
        config.num_inference_steps != 20
        or config.video_exec_step != -1
        or config.action_num_inference_steps != 50
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
