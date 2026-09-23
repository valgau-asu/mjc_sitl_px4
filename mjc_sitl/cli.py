"""Command line interface."""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _contract_from_args(a):
    from .contract import DeployContract, Tolerances
    kw = dict(name=a.name or a.controller, controller_kind=a.controller,
              task=a.task, vehicle=a.vehicle, scene=a.scene, zone=a.zone,
              control_hz=a.control_hz, duration=a.duration,
              obstacles=not a.no_obstacles, planner=a.planner)
    if a.controller == "custom":
        kw.update(controller_path=str(Path(a.controller_path).resolve()),
                  controller_class=a.controller_class)
    if a.policy:
        kw["policy_path"] = str(Path(a.policy).resolve())
    if a.pos_tol:
        kw["tolerances"] = Tolerances(max_pos_rmse=a.pos_tol)
    return DeployContract(**kw)


def _add_contract_args(p):
    p.add_argument("--controller", default="mine",
                   choices=["mine", "pd", "mpc", "ppo", "custom"],
                   help="which sim_mjc controller to wrap (default: mine)")
    p.add_argument("--controller-path", default=None,
                   help="source file, for --controller custom")
    p.add_argument("--controller-class", default="MyController")
    p.add_argument("--policy", default=None, help="checkpoint, for --controller ppo")
    p.add_argument("--task", default="hover", choices=["hover", "figure8"])
    p.add_argument("--vehicle", default="sim_default", choices=["sim_default", "dart"])
    p.add_argument("--scene", default="generated", choices=["generated", "dart"],
                   help="'generated' shares sim_mjc's compiled model (default)")
    p.add_argument("--zone", default="D")
    p.add_argument("--planner", default="inline", choices=["inline", "node"])
    p.add_argument("--control-hz", type=int, default=50)
    p.add_argument("--duration", type=float, default=12.0)
    p.add_argument("--no-obstacles", action="store_true")
    p.add_argument("--name", default=None)
    p.add_argument("--pos-tol", type=float, default=None,
                   help="max sim-vs-SITL position RMSE [m]")


def _artifacts(contract, sim, sitl, outdir, a, tier=""):
    """Render comparison figures and video. Never raises."""
    if a.no_plots and a.no_video:
        return []
    from . import viz
    h = a.video_height
    print(f"\nartifacts{f' ({tier} tier)' if tier else ''} -> {outdir}")
    if tier == "fast" and not a.no_video:
        print("  note: the fast tier emulates the deployment in-process, so "
              "the video is not the ROS 2 run -- use --tier full for that")
    off = tuple(getattr(a, "ghost_offset", (0.0, 0.0, 0.0)))
    if any(off):
        print(f"  ghost displaced by {off} m in the video for visibility -- "
              f"the figures and metrics are unshifted")
    paths = viz.save_artifacts(contract, sim, sitl, outdir,
                               video=not a.no_video, fps=a.fps,
                               size=(h, int(round(h * 16 / 9))), verbose=True,
                               ghost_offset=off)
    if not a.no_video and not any(str(x).endswith(".mp4") for x in paths):
        print("  (no video was written -- see the reason above; if it mentions "
              "GL, try MUJOCO_GL=egl)")
    return paths


def _add_artifact_args(p):
    p.add_argument("--no-video", action="store_true",
                   help="skip the .mp4 renders (they dominate the runtime)")
    p.add_argument("--no-plots", action="store_true",
                   help="skip the sim-vs-SITL comparison figures")
    p.add_argument("--fps", type=int, default=30, help="video frame rate")
    p.add_argument("--video-height", type=int, default=720,
                   help="video height in pixels; width follows 16:9")
    p.add_argument("--ghost-offset", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                   metavar=("X", "Y", "Z"),
                   help="displace the sim ghost in the video by these metres, "
                        "purely to view the two drones side by side. The "
                        "figures and every metric stay unshifted.")


def cmd_compare(a):
    """Fly sim and SITL, compare, render."""
    from .contract import DeployContract
    from .validators import dynamics, safety, summarize

    if a.from_dir:
        d = Path(a.from_dir)
        contract = DeployContract.load(d / "deploy_contract.json")
        adapter = d / "adapter.py"
        outdir = Path(a.outdir) if a.outdir else d
    elif a.contract:
        contract = DeployContract.load(a.contract)
        adapter = Path(a.adapter or Path(a.contract).parent / "adapter.py")
        outdir = Path(a.outdir or Path(a.contract).parent)
    else:
        contract = _contract_from_args(a)
        adapter = Path(a.adapter) if a.adapter else (
            Path(__file__).resolve().parent / "offboard_lib" / "adapter.py")
        outdir = Path(a.outdir or REPO_ROOT / "out" / "compare" / contract.name)
    if not Path(adapter).exists():
        print(f"no adapter at {adapter}")
        return 1
    outdir.mkdir(parents=True, exist_ok=True)

    print(contract.summary())
    print(f"adapter: {adapter}\ntier:    {a.tier}\n")

    print("flying the sim_mjc reference ...")
    sim = dynamics.run_sim_reference(contract, adapter)
    print(f"  {len(sim['t'])} samples, rmse {sim['metrics'].get('rmse', float('nan')):.4f} m")

    print(f"flying SITL ({a.tier}) ...")
    sitl = (dynamics.run_fast_sitl(contract, adapter) if a.tier == "fast"
            else dynamics.run_full_sitl(contract, adapter, outdir,
                                        rate_scale=a.rate_scale))
    if a.tier == "full" and a.rate_scale != 1.0:
        print(f"  NOTE: ran at {a.rate_scale}x wall-clock -- a diagnostic, not "
              f"a passing deployment")
    if sitl.get("ok") is False:
        print(f"  SITL failed: {sitl.get('error')}")
        return 1
    print(f"  {len(sitl['t'])} samples")

    res = dynamics.compare(sim, sitl, contract, a.tier)
    print()
    print(summarize([res, safety.validate(sitl, contract)]))
    _artifacts(contract, sim, sitl, outdir, a, tier=a.tier)
    return 0 if res.passed else 1


def cmd_run(a):
    """Fly one controller through ROS 2 SITL and record it."""
    from .run_sitl import fly
    contract = _contract_from_args(a)
    adapter = (Path(a.adapter).resolve() if a.adapter
               else Path(__file__).resolve().parent / "offboard_lib" / "adapter.py")
    outdir = Path(a.outdir or (REPO_ROOT / "demo" /
                               f"{a.controller}_{a.task}"))
    print(contract.summary(), "\n")
    fly(contract, adapter, outdir, video=not a.no_video, rate_scale=a.rate_scale,
        fps=a.fps, size=(a.video_height, int(a.video_height * 16 / 9) // 2 * 2))
    return 0


def cmd_parity(a):
    from .contract import DeployContract
    from . import scenegen
    c = DeployContract(scene=a.scene, vehicle=a.vehicle, zone=a.zone)
    print(c.spec.summary())
    print()
    print(scenegen.describe(c))
    rep = scenegen.verify_shared_physics(c)
    print(f"\nparity: {'OK' if rep['ok'] else 'PROBLEMS'} "
          f"({sum(rep['checks'].values())}/{len(rep['checks'])} checks)")
    if not rep["ok"]:
        for p in rep["problems"]:
            print(f"  {p}")
    return 0 if rep["ok"] else 1


def cmd_check(a):
    from .contract import DeployContract
    from .validators import coding, dynamics, safety, summarize
    contract = DeployContract.load(a.contract)
    print(contract.summary(), "\n")
    results = [coding.validate(a.adapter, contract)]
    if results[0].passed:
        tier = a.tier
        run = (dynamics.run_fast_sitl(contract, a.adapter) if tier == "fast"
               else dynamics.run_full_sitl(contract, a.adapter,
                                           a.workdir or "/tmp/mjc_sitl_check"))
        sim = dynamics.run_sim_reference(contract, a.adapter)
        results.append(dynamics.compare(sim, run, contract, tier))
        results.append(safety.validate(run, contract))
    print(summarize(results))
    ok = all(r.passed for r in results)
    print(f"\n{'ALL GATES PASS' if ok else 'GATES FAILED'}")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="mjc_sitl", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="fly a controller through ROS 2 SITL")
    _add_contract_args(p)
    p.add_argument("--adapter", default=None)
    p.add_argument("--outdir", default=None)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--video-height", type=int, default=480)
    p.add_argument("--rate-scale", type=float, default=1.0,
                   help="sim clock speed. <1 slows it so a heavy controller "
                        "(e.g. MPC) can keep up; the control loop is unchanged")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("parity", help="show whether sim and SITL share physics")
    p.add_argument("--scene", default="generated", choices=["generated", "dart"])
    p.add_argument("--vehicle", default="sim_default", choices=["sim_default", "dart"])
    p.add_argument("--zone", default="D")
    p.set_defaults(fn=cmd_parity)


    p = sub.add_parser("compare",
                       help="fly sim and SITL and render the comparison")
    _add_contract_args(p)
    _add_artifact_args(p)
    p.add_argument("--from", dest="from_dir", default=None, metavar="DIR",
                   help="reuse a previous deployment's contract and adapter, "
                        "e.g. out/deploy/pd")
    p.add_argument("--contract", default=None)
    p.add_argument("--adapter", default=None)
    p.add_argument("--tier", default="full", choices=["fast", "full"],
                   help="'full' runs the real ROS 2 stack (default)")
    p.add_argument("--rate-scale", type=float, default=1.0,
                   help="run SITL slower than wall-clock (e.g. 0.25) to tell a "
                        "wrong control law from one that is merely too slow")
    p.add_argument("--outdir", default=None)
    p.set_defaults(fn=cmd_compare)


    a = ap.parse_args(argv)
    if a.cmd == "deploy" and a.controller == "custom" and not a.controller_path:
        ap.error("--controller custom requires --controller-path")
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
