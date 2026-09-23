# mjc_sitl_px4

Three quadrotor controllers — **PD, linear MPC, and a trained PPO policy** — each
flown as a **ROS 2 node** against MuJoCo, with a video and flight data for each.

```bash
./demo.sh show     # open the recorded flights          (instant)
./demo.sh pd       # fly PD through ROS 2 SITL          (~60 s)
./demo.sh mpc      # fly MPC                            (~150 s)
./demo.sh ppo      # fly the trained PPO policy         (~60 s)
```

## The three flights

Figure-8, 24 s, through the real ROS 2 stack. Each writes `flight.mp4` and
`flight.png` into `demo/<controller>_figure8/`.

| controller | tracking RMSE | control solve (p95) | note |
|---|---|---|---|
| **PD** | 0.119 m | 0.09 ms | cascaded position → tilt → torque |
| **MPC** | **0.078 m** | 8.3 ms | convex QP on the hover linearisation |
| **PPO** | 0.133 m | 0.43 ms | residual policy on top of the PD baseline |

MPC tracks best and costs ~90× more compute than PD. PPO is worst here, which is
expected: with no wind, PD is already close to optimal on this task and a residual
policy mostly adds exploration noise.

### MPC needs a slower clock — and that is the interesting part

At 50 Hz control and a 5 ms physics step, MPC **cannot keep up in wall-clock time**
and the vehicle diverges within 20 steps. The same controller is perfectly stable
in-process. So `./demo.sh mpc` runs the sim clock at `--rate-scale 0.2`: the control
loop is unchanged, the wall-clock run just takes 5× longer.

That is a real deployment constraint, not a bug — and it is exactly the kind of thing
that only shows up once a controller is a real node with a real deadline.

## How it works

The ROS 2 node runs the controller against a MuJoCo model built by `sim_mjc`'s own
`build_scene()`, so the simulation and the deployed node compile the **same model**.

```
$ python3 -m mjc_sitl parity
parity: OK (12/12 checks)
```

Masses, inertias, gearing, thrust limits, timestep and the allocation matrix, all
compared between the two compiled models. That is what makes a sim-vs-SITL comparison
meaningful — `python3 -m mjc_sitl compare` uses it.

Two nodes, in `mjc_sitl/offboard_lib/`:

* `sim_node` — steps MuJoCo, publishes `VehicleOdometry`, holds physics until the
  first command so a run is deterministic regardless of startup order
* `controller_node` — subscribes, runs the control law, publishes four rotor thrusts

## What it cannot do

**This is not flight-ready PX4 code, and PX4 is not running.** Only the message
*types* are borrowed from `px4_msgs`; commands are four thrusts in **newtons** on a
custom topic. To fly Gazebo/PX4 SITL or hardware it needs:

| missing | consequence |
|---|---|
| NED ↔ ENU conversion | this stack is z-**up** and hovers at `position[2] = +1.0`; PX4 odometry is z-**down**. The altitude loop is sign-inverted — it flies into the ground |
| `OffboardControlMode` heartbeat | PX4 drops offboard mode without it at >2 Hz |
| `VehicleCommand` arm / mode | nothing arms the vehicle |
| newtons → `ActuatorMotors` | PX4 wants normalised [0,1], calibrated per airframe |
| takeoff / land / failsafe | no mode-change or RC-override handling |

Nothing here models sensor noise, state estimation (odometry is ground truth, not an
EKF), actuator lag or wind. These flights show the controllers **work as ROS 2 nodes**
— they say nothing about robustness.

## Where it goes next

```
1. Develop        2. Runs as a      3. MuJoCo          4. Hardware
   in MuJoCo  -->    ROS 2 node  -->   PX4 SITL   -->     the real
   [done]            [done]            [open]             vehicle
```

Stage 3 is a **MuJoCo surrogate for Gazebo PX4 SITL**. Gazebo would give the PX4
protocol but a *different physics engine* — and once the engine differs you can no
longer separate a deployment bug from an engine difference. Doing it in MuJoCo keeps
the protocol *and* the shared model.

Order: frame conversion (with a round-trip test) → offboard heartbeat and arming →
thrust normalisation → run against real PX4 SITL → takeoff/land/failsafe. None of it
is algorithmically hard; the failures are just **silent**, which is why it wants tests.

## Requirements

Python 3.10, ROS 2 Humble, `mujoco>=3.3`, `numpy`, `scipy`, and `px4_msgs` for the
message types. `cvxpy` for MPC, `torch` for PPO, `matplotlib` + `mediapy` for output.

```bash
export PX4_WS=/path/to/your/px4_ws     # where px4_msgs is built
source setup_env.sh
```

Nothing to build — the SITL nodes run as plain Python processes. Headless? Prefix with
`MUJOCO_GL=egl`.

## Layout

```
sim_mjc/              the controllers (PD, MPC, PPO) and the vehicle definition
mjc_sitl/             the ROS 2 nodes, the shared scene, the comparison tools
mujoco_px4_upstream/  the original SITL stack -- see its ATTRIBUTION.md
demo/                 the three recorded flights
docs/                 slides
```

## Credits

`mujoco_px4_upstream/` is vendored from <https://github.com/YogeshMaan/mujoco_px4>,
© Yogesh Maan — the original MuJoCo SITL stack this grew out of. See
`mujoco_px4_upstream/ATTRIBUTION.md`.
