"""A trained RL policy, wrapped as an ordinary controller.

Once wrapped, a policy goes through `Simulator.run()` like PD or MPC — same scorecard,
same plots, same video. That is the point: one comparison, three families of method.
"""
import numpy as np

from ..rl.env import DroneEnv, EnvConfig, Observer
from .base import Controller


class PolicyController(Controller):
    """Wrap `policy(obs) -> action` in the controller interface.

    `env` supplies the action decoding and observation layout, so the policy sees at
    run time exactly what it saw during training.
    """

    def __init__(self, policy, env: DroneEnv, name="PPO"):
        super().__init__(env.traj, name)
        self.policy, self.env = policy, env
        cfg = env.cfg
        self.observer = Observer(env.traj, cfg.preview, cfg.preview_dt,
                                 env.action_space.shape[0])

    @classmethod
    def from_checkpoint(cls, path, cfg: EnvConfig = None, name="PPO"):
        """Load a policy saved by `sim_mjc.rl.ppo.save()`."""
        from ..rl.ppo import load
        policy, ck = load(str(path))
        if cfg is None:
            stored = dict(ck["cfg"])
            stored.pop("traj", None)
            cfg = EnvConfig(**stored)
        return cls(policy, DroneEnv(cfg), name), ck

    @classmethod
    def from_sb3(cls, model, env, name="PPO (SB3)", deterministic=True):
        def predict(obs):
            action, _ = model.predict(obs, deterministic=deterministic)
            return action
        return cls(predict, env, name)

    def reset(self):
        self.observer.reset()

    def control(self, t, state):
        obs = self.observer.build(t, state.pos, state.vel, state.quat, state.omega)
        action = np.clip(np.asarray(self.policy(obs), float).ravel(), -1.0, 1.0)
        self.observer.prev_action = action
        return self.env.action_to_thrusts(action, state, t)
