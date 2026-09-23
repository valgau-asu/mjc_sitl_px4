"""A compact PPO: actor-critic MLP, GAE, clipped surrogate.

Enough to train a good tracking policy on this simulator in a few minutes of CPU, and
short enough to read. The environment is a standard Gymnasium environment, so
Stable-Baselines3 or any other library will drive it unchanged if you prefer.
"""
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

from .env import DroneEnv, EnvConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------------------
# Running observation normalization
# --------------------------------------------------------------------------------------
class RunningNorm:
    """Running mean/variance over observations, saved with the weights."""

    def __init__(self, dim):
        self.mean = np.zeros(dim, np.float64)
        self.var = np.ones(dim, np.float64)
        self.count = 1e-4

    def update(self, x):
        x = np.asarray(x, np.float64).reshape(-1, self.mean.size)
        n = x.shape[0]
        if n == 0:
            return
        mean, var = x.mean(0), x.var(0)
        delta = mean - self.mean
        tot = self.count + n
        self.mean += delta * n / tot
        m_a = self.var * self.count
        m_b = var * n
        self.var = (m_a + m_b + delta ** 2 * self.count * n / tot) / tot
        self.count = tot

    def __call__(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8),
                       -10, 10).astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d):
        self.mean, self.var, self.count = d["mean"], d["var"], d["count"]


# --------------------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------------------
def mlp(sizes, act=nn.Tanh):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1])]
        if i < len(sizes) - 2:
            layers += [act()]
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(64, 64), init_log_std=-1.5):
        super().__init__()
        self.pi = mlp([obs_dim, *hidden, act_dim])
        self.vf = mlp([obs_dim, *hidden, 1])
        # -0.5 is the usual default; on this vehicle that much exploration noise
        # tumbles it on the first rollout.
        self.log_std = nn.Parameter(init_log_std * torch.ones(act_dim))
        # small final layer -> initial policy is ~zero action, i.e. the PD baseline
        with torch.no_grad():
            self.pi[-1].weight.mul_(0.01)
            self.pi[-1].bias.mul_(0.0)

    def dist(self, obs):
        return torch.distributions.Normal(self.pi(obs), self.log_std.exp())

    def act(self, obs):
        with torch.no_grad():
            d = self.dist(obs)
            a = d.sample()
            return (a.cpu().numpy(),
                    d.log_prob(a).sum(-1).cpu().numpy(),
                    self.vf(obs).squeeze(-1).cpu().numpy())

    def value(self, obs):
        with torch.no_grad():
            return self.vf(obs).squeeze(-1).cpu().numpy()


class TorchPolicy:
    """Callable `policy(obs) -> action`, which is what PolicyController expects."""

    def __init__(self, net: ActorCritic, norm: RunningNorm, deterministic=True):
        self.net, self.norm, self.deterministic = net, norm, deterministic

    def __call__(self, obs):
        x = torch.as_tensor(self.norm(obs), dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            mu = self.net.pi(x.unsqueeze(0))
            if not self.deterministic:
                mu = mu + self.net.log_std.exp() * torch.randn_like(mu)
        return mu.squeeze(0).cpu().numpy()


def save(path, net, norm, cfg, history=None):
    torch.save({"net": net.state_dict(), "norm": norm.state_dict(),
                "cfg": {k: v for k, v in asdict(cfg).items() if k != "traj"},
                "obs_dim": net.pi[0].in_features,
                "act_dim": net.log_std.numel(),
                "history": history}, path)


def load(path):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    net = ActorCritic(ck["obs_dim"], ck["act_dim"]).to(DEVICE)
    net.load_state_dict(ck["net"])
    net.eval()
    norm = RunningNorm(ck["obs_dim"])
    norm.load_state_dict(ck["norm"])
    return TorchPolicy(net, norm), ck


# --------------------------------------------------------------------------------------
# PPO
# --------------------------------------------------------------------------------------
@dataclass
class Rollout:
    """One batch of experience, shaped (horizon, n_envs, ...)."""
    obs:  np.ndarray
    act:  np.ndarray
    logp: np.ndarray
    val:  np.ndarray
    rew:  np.ndarray
    term: np.ndarray        # a real terminal state (crash)
    boot: np.ndarray        # value of the final state where the time limit cut in
    cut:  np.ndarray        # where that happened


class History:
    """Per-iteration training statistics, for the learning curve.

    Mean and standard deviation are taken over the last `window` completed episodes
    rather than over the handful that finish inside one iteration, which would be far
    too noisy to read.
    """

    def __init__(self, window=20):
        self.window = window
        self.returns, self.scores = [], []
        self.log = {k: [] for k in ("iteration", "steps", "return_mean", "return_std",
                                    "score_mean", "score_std", "episodes")}

    def add_episode(self, ret, score):
        self.returns.append(ret)
        if np.isfinite(score):
            self.scores.append(score)

    def record(self, iteration, steps):
        r, s = self.returns[-self.window:], self.scores[-self.window:]
        self.log["iteration"].append(iteration)
        self.log["steps"].append(steps)
        self.log["return_mean"].append(float(np.mean(r)) if r else np.nan)
        self.log["return_std"].append(float(np.std(r)) if len(r) > 1 else 0.0)
        self.log["score_mean"].append(float(np.mean(s)) if s else np.nan)
        self.log["score_std"].append(float(np.std(s)) if len(s) > 1 else 0.0)
        self.log["episodes"].append(len(self.returns))

    def as_dict(self):
        return {k: np.asarray(v) for k, v in self.log.items()}


def _collect(envs, net, norm, horizon, obs, running, hist):
    """Step every env `horizon` times and return the batch."""
    n, od, ad = len(envs), obs.shape[1], envs[0].action_space.shape[0]
    z = lambda *shape: np.zeros(shape, np.float32)
    roll = Rollout(z(horizon, n, od), z(horizon, n, ad), z(horizon, n), z(horizon, n),
                   z(horizon, n), z(horizon, n), z(horizon, n),
                   np.zeros((horizon, n), bool))

    for t in range(horizon):
        norm.update(obs)
        nobs = norm(obs)
        a, lp, v = net.act(torch.as_tensor(nobs, device=DEVICE))
        roll.obs[t], roll.act[t], roll.logp[t], roll.val[t] = nobs, a, lp, v

        for i, env in enumerate(envs):
            o2, r, term, trunc, info = env.step(a[i])
            roll.rew[t, i], roll.term[t, i] = r, float(term)
            running[i] += r

            if term or trunc:
                # hitting the time limit is not a real terminal state: bootstrap off
                # the final value instead of treating it as one
                if trunc and not term:
                    roll.cut[t, i] = True
                    roll.boot[t, i] = net.value(
                        torch.as_tensor(norm(o2)[None], device=DEVICE))[0]
                hist.add_episode(running[i], info.get("episode_score", np.inf))
                running[i] = 0.0
                o2, _ = env.reset()
            obs[i] = o2
    return roll


def _advantages(roll, last_v, gamma, lam):
    """Generalized advantage estimation."""
    adv = np.zeros_like(roll.rew)
    gae = np.zeros(roll.rew.shape[1], np.float32)
    for t in reversed(range(len(roll.rew))):
        nonterminal = 1.0 - roll.term[t]
        next_v = last_v if t == len(roll.rew) - 1 else roll.val[t + 1]
        next_v = np.where(roll.cut[t], roll.boot[t], next_v)
        delta = roll.rew[t] + gamma * next_v * nonterminal - roll.val[t]
        gae = delta + gamma * lam * nonterminal * np.where(roll.cut[t], 0.0, gae)
        adv[t] = gae
    return adv, adv + roll.val


def _optimize(net, opt, roll, adv, ret, epochs, minibatches,
              clip, vf_coef, ent_coef, max_grad):
    """The clipped-surrogate update."""
    flat = lambda x, d=None: torch.as_tensor(
        x.reshape(-1, d) if d else x.reshape(-1), device=DEVICE)
    b_obs, b_act = flat(roll.obs, roll.obs.shape[-1]), flat(roll.act, roll.act.shape[-1])
    b_lp, b_ret = flat(roll.logp), flat(ret)
    b_adv = flat(adv)
    b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

    batch = b_obs.shape[0]
    idx = np.arange(batch)
    mb = batch // minibatches
    for _ in range(epochs):
        np.random.shuffle(idx)
        for start in range(0, batch, mb):
            j = torch.as_tensor(idx[start:start + mb], device=DEVICE)
            d = net.dist(b_obs[j])
            ratio = (d.log_prob(b_act[j]).sum(-1) - b_lp[j]).exp()
            pg = -torch.min(ratio * b_adv[j],
                            ratio.clamp(1 - clip, 1 + clip) * b_adv[j]).mean()
            vloss = ((net.vf(b_obs[j]).squeeze(-1) - b_ret[j]) ** 2).mean()
            loss = pg + vf_coef * vloss - ent_coef * d.entropy().sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_grad)
            opt.step()


def train(cfg: EnvConfig, total_steps=400_000, n_envs=8, horizon=256,
          epochs=10, minibatches=8, gamma=0.99, lam=0.95, clip=0.2,
          lr=3e-4, ent_coef=0.0, vf_coef=0.5, max_grad=0.5, seed=0, log_every=1):
    """Train a policy on `cfg`. Returns (net, norm, env, history)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    envs = [DroneEnv(cfg) for _ in range(n_envs)]
    obs_dim = envs[0].observation_space.shape[0]
    net = ActorCritic(obs_dim, envs[0].action_space.shape[0]).to(DEVICE)
    norm = RunningNorm(obs_dim)
    opt = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-5)

    obs = np.stack([e.reset(seed=seed + i)[0] for i, e in enumerate(envs)])
    running = np.zeros(n_envs)
    hist = History()
    batch = n_envs * horizon
    n_updates = max(1, total_steps // batch)
    t0 = time.perf_counter()

    for update in range(1, n_updates + 1):
        roll = _collect(envs, net, norm, horizon, obs, running, hist)
        last_v = net.value(torch.as_tensor(norm(obs), device=DEVICE))
        adv, ret = _advantages(roll, last_v, gamma, lam)
        _optimize(net, opt, roll, adv, ret, epochs, minibatches,
                  clip, vf_coef, ent_coef, max_grad)
        hist.record(update, update * batch)

        if update % log_every == 0 or update == n_updates:
            L = hist.log
            print(f"  update {update:3d}/{n_updates}  steps {update * batch:>7,}  "
                  f"return {L['return_mean'][-1]:8.2f} +/- {L['return_std'][-1]:5.2f}  "
                  f"score {L['score_mean'][-1]:7.4f}  "
                  f"{update * batch / (time.perf_counter() - t0):5.0f} steps/s",
                  flush=True)

    return net, norm, envs[0], hist.as_dict()


def plot_training(history, figsize=(9, 5)):
    """Learning curve: mean +/- one standard deviation over recent episodes."""
    import matplotlib.pyplot as plt

    h = {k: np.asarray(v) for k, v in history.items()}
    it = h["iteration"]
    fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=True)

    for a, mean, std, label, color in (
            (ax[0], h["return_mean"], h["return_std"], "episode return", "#1f77b4"),
            (ax[1], h["score_mean"], h["score_std"], "benchmark score", "#d62728")):
        a.plot(it, mean, color=color, lw=1.8)
        a.fill_between(it, mean - std, mean + std, color=color, alpha=0.20, lw=0)
        a.set_ylabel(label)
        a.grid(alpha=0.3)

    # the score is positive by construction, so clip the band there rather than
    # letting mean - std run below the axis
    ax[1].set_ylim(bottom=0.0)
    ax[1].set_xlabel("training iteration")
    ax[0].set_title("mean $\\pm$ 1 s.d. over the last 20 episodes")
    fig.tight_layout()
    return fig


