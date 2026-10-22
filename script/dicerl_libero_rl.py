import wandb

from agent.finetune.train_distill_residual_flow_img_agent import TrainDistillResidualFlowImgAgent
from env.gym_utils.wrapper.libero_shift import ramp_magnitude


class ShiftedResidualAgent(TrainDistillResidualFlowImgAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.venv.call_sync_arg("set_shift_seed", "seed", [cfg.shift.seed + i for i in range(cfg.env.n_envs)])
        self._shift_magnitude = None
        self._shift_logged_step = -1

    def update_networks(self, training_step=0):
        shift = self.cfg.shift
        magnitude = ramp_magnitude(training_step, shift.ramp_start, shift.ramp_end, shift.end_magnitude)
        if magnitude != self._shift_magnitude:
            self.venv.call_sync("set_shift", magnitude=magnitude)
            self._shift_magnitude = magnitude
        if training_step != self._shift_logged_step and self.use_wandb:
            self._shift_logged_step = training_step
            stats = [value for value, _ in self.venv.call_sync("shift_stats")]
            wandb.log(
                {"shift/magnitude": magnitude, **{f"shift/{key}": sum(s[key] for s in stats) for key in ("draws", "rejections", "fallbacks")}},
                step=training_step * self.horizon_steps * self.n_envs,
            )
        return super().update_networks(training_step)
