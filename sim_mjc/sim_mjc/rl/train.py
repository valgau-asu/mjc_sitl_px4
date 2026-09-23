"""Train a PPO policy on one of the standard tasks.

    python -m sim_mjc.rl.train --task figure8 --steps 300000
    python -m sim_mjc.rl.train --task hover --steps 200000
"""
import argparse
from pathlib import Path

import numpy as np

from .env import DroneEnv, evaluate, task_config
from .ppo import TorchPolicy, save, train

DEFAULT_OBSTACLES = [
    {"type": "pillar", "pos": (0.0, 2.6, 1.0)},
    {"type": "pillar", "pos": (0.0, -2.6, 1.0)},
    {"type": "box", "pos": (3.4, 2.2, 0.5)},
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="figure8", choices=["hover", "figure8"])
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--action-mode", default="residual",
                    choices=["residual", "wrench", "thrusts"])
    ap.add_argument("--no-obstacles", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--curve", metavar="PATH", default=None,
                    help="save the learning curve to PATH")
    args = ap.parse_args(argv)

    cfg = task_config(args.task, action_mode=args.action_mode,
                      obstacles=() if args.no_obstacles else DEFAULT_OBSTACLES)
    out = Path(args.out or f"policies/ppo_{args.task}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)

    probe = DroneEnv(cfg)
    base = evaluate(lambda o: np.zeros(4), probe, episodes=5)
    print(f"PD baseline (zero action): score {base['score']:.4f}  "
          f"rmse {base['rmse']:.3f} m\n")

    net, norm, env, history = train(cfg, total_steps=args.steps,
                                    n_envs=args.envs, seed=args.seed)
    save(out, net, norm, cfg, history)
    print(f"\nsaved {out}")

    final = evaluate(TorchPolicy(net, norm), env, episodes=5)
    print(f"PD  : score {base['score']:.4f}  rmse {base['rmse']:.3f}")
    print(f"PPO : score {final['score']:.4f}  rmse {final['rmse']:.3f}  "
          f"crashes {final['crashes']}")

    if args.curve:
        from .ppo import plot_training
        plot_training(history).savefig(args.curve, dpi=150, bbox_inches="tight")
        print(f"wrote {args.curve}")
    return net, norm, history


if __name__ == "__main__":
    main()
