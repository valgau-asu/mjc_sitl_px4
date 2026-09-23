# CLAUDE_SIM.MD — how sim_mjc runs

Orientation for anyone (or any agent) working on this package. `README.md` is the user
guide; this file explains the machinery.

---

## 1. What this is

A MuJoCo quadrotor simulator built around a single interface:

```python
control(t, state) -> np.ndarray   # four rotor thrusts in newtons, each in [0, 7]
```

Four controllers implement it — `PDController`, `LinearMPC`, `PolicyController` (a
trained RL policy) and `MyController` (a template). Two standard tasks: `hover`
(setpoint) and `figure8`. There is **no wind**; a run is deterministic given a
controller and an initial condition.

The package descends from a teaching notebook, but nothing here depends on it.

---

## 2. Running it

```bash
python -m sim_mjc.cli --controller pd   --task hover
python -m sim_mjc.cli --controller mpc  --task figure8 --plot --video
python -m sim_mjc.cli --controller mine --task figure8
python -m sim_mjc.cli --compare --task figure8 --policy policies/ppo_figure8.pt

python -m sim_mjc.rl.train --task figure8 --steps 300000
pytest tests/                      # 9 tests, ~10 s
```

Equivalent scripts live in `examples/`. By default they write a scorecard, three
figures and a 720p chase-camera video per task into `out/<controller>_<task>/`:

```bash
python3 examples/run_pd.py                 # both tasks, plots + video
python3 examples/run_pd.py --no-video      # much faster; rendering dominates
python3 examples/run_pd.py --show          # pop the figures up too
```

---

## 3. The normal control flow

`Simulator` owns the loop and calls **into** the controller.

```
cli.main()
  └─ build_controller(kind, traj)          -> a Controller
  └─ Simulator(obstacles=..., control_hz=50)
        └─ model.build_scene()             -> compiled MjModel   (once per Simulator)
  └─ Simulator.run(controller, duration)
        ├─ reset(start)                    -> mj_resetData + initial qpos
        └─ for k in range(duration / dt):          # dt = 0.005 s, 200 Hz
             state = self.state()                  # DroneState, incl. one IMU sample
             if k % decim == 0:                    # decim = 4  ->  50 Hz
                 u = controller.control(t, state)  # <- your code
                 u = clip(u, 0, MAX_THRUST)        # actuators saturate
             data.ctrl[:] = u                      # zero-order hold in between
             mujoco.mj_step(model, data)
             log position, velocity, attitude, thrusts, reference, IMU
             detect contact; check crash
             every 1/video_fps: grab camera frames
        └─ compute_metrics(...)            -> SimResult.metrics
```

Two rates matter and are easy to confuse:

| | |
|---|---|
| **physics** | 200 Hz, `model.opt.timestep = 0.005` |
| **control** | `control_hz`, default 50 — must divide 200 |
| `decim` | `(1/control_hz) / dt`, so 4 at the defaults |

The command is held constant between control steps. That zero-order hold is what makes
the MPC solve-time budget meaningful: `metrics["solve_ms_p95"]` against `1000/control_hz`.

**Return value.** `run()` gives a `SimResult` with `t, pos, vel, euler, omega, thrusts,
ref, metrics, frames, fpv, depth, imu`. `euler` and `omega` are in **degrees** for
plotting; the `DroneState` handed to a controller is in **radians**.

---

## 4. The RL control flow — inverted

Reinforcement learning needs the opposite arrangement: the agent owns the loop.

```
rl.train.main()
  └─ task_config(task)                     -> EnvConfig
  └─ rl.ppo.train(cfg)
        ├─ [DroneEnv(cfg) for _ in range(n_envs)]
        └─ per iteration:
             _collect()      -> Rollout      # env.step(action), n_envs in lockstep
             _advantages()   -> GAE
             _optimize()     -> clipped surrogate update
             History.record()                # for the learning curve
  └─ save(path, net, norm, cfg, history)
```

`DroneEnv` is a separate implementation from `Simulator`, deliberately: it compiles its
model once, renders nothing, logs nothing, and reaches ~2,200 steps/s per process. It
shares the *model, dynamics and cost*, not the loop.

`PolicyController` closes the circle — it wraps a trained policy back into
`control(t, state)`, so an RL policy is scored by `Simulator` exactly like PD and MPC.
`Observer` builds the policy's input and is shared by `DroneEnv` and `PolicyController`
**so the two cannot drift apart**; a mismatch there is the classic reason a policy that
trained well evaluates like noise.

---

## 5. Module map

| module | holds | depends on |
|---|---|---|
| `config.py` | `DRONE` dict, derived constants, `ZONE_OFFSETS`, `OBSTACLE_LIB` | numpy |
| `model/xml.py` | MJCF text for the drone, arena and brick texture | config |
| `model/scene.py` | assembles and compiles the scene **in memory** | model/xml |
| `dynamics.py` | `MIXER`, `ALLOCATION`, `linearize()`, `discretize()`, state indices | config |
| `sensors.py` | `DroneState`, `IMU`, `SensorSuite`, `Camera` | dynamics |
| `simulator.py` | `Simulator`, `SimResult`, `simulate()` | most of the above |
| `trajectories.py` | `setpoint`, `figure_eight`, heading helpers, `TASKS` | dynamics |
| `metrics.py` | `compute_metrics`, `scorecard`, `compare` | config |
| `viz.py` | figures (returned, never shown), `save_plots`, `save_video` | config (matplotlib optional) |
| `presets.py` | default course, episode durations, `build_controller` | controllers |
| `report.py` | `run_and_report` — the one place a run's outputs are decided | presets, simulator, viz |
| `cli.py` | one entry point for every controller and task | everything |
| `controllers/` | `base`, `pd`, `mpc`, `ppo`, `my_controller` | dynamics, trajectories |
| `rl/` | `env` (Gymnasium), `ppo` (trainer), `train` (CLI) | model, dynamics |

Dependencies point downward only; nothing in `config`/`model`/`dynamics` imports a
controller. The one deliberate exception is `rl/env.py`, which imports `cascaded_pd`
*inside* `action_to_thrusts` — residual actions are defined relative to the PD baseline,
and a module-level import there would be circular.

**Optional dependencies.** `controllers/__init__.py` imports `LinearMPC` and
`PolicyController` in `try/except ImportError` and sets them to `None` if cvxpy or torch
is missing. PD must always work on a bare `mujoco + numpy + scipy` install.

---

## 6. Data structures

```python
DroneState(t, pos, vel, quat, euler, omega, qpos, qvel, imu)
```

`pos` is **zone-local**: the middle of the flight zone is always the origin, whichever
zone you fly in. `ZONE_OFFSETS[zone]` converts to world coordinates, and the simulator
subtracts it on the way out and adds it on the way in.

`state.imu` is the honest channel — five MuJoCo sensors read from `data.sensordata`
rather than from ground truth:

| field | sensor | note |
|---|---|---|
| `gyro` | `gyro` | body angular rate [rad/s] |
| `accel` | `accelerometer` | **proper** acceleration — reads `[0, 0, 9.81]` at hover, not zero |
| `quat` | `framequat` | `(w, x, y, z)` |
| `pos` | `framepos` | offset-corrected to zone-local |
| `linvel` | `framelinvel` | world frame |

`SensorSuite` resolves sensor addresses once at construction; if a model carries no
sensors, `read()` returns `None` and `state.imu` is `None`.

---

## 7. Things that will bite you

**The offscreen framebuffer.** MuJoCo renders cameras into a fixed buffer that defaults
to 640×480, and asking for a larger image raises. `build_scene()` takes `offwidth`/
`offheight` and writes them into the MJCF `<visual><global>` clause; `Simulator` sizes
them from `video_size` and `fpv_size`. Raise them if you want larger renders.

**Scene assembly is in memory.** `build_scene()` passes the XML as a string with the
static includes supplied as an assets dict. Do not "simplify" this into writing
`scene.xml` to disk — several simulators or RL workers would then race over one file.

**`control_hz` must divide 200.** Checked in `Simulator.__init__`. Valid: 200, 100, 50,
40, 25, 20, 10.

**Yaw wraps.** Use `wrap_angle()` on every yaw error. In the RL observation, attitude
enters as rotation-matrix columns rather than Euler angles for exactly this reason.

**Residual action scaling.** `EnvConfig.residual_scale` is sized against the *rotational*
inertia, not the thrust range: `Ixx = 0.0025 kg·m²`, so a differential pair of 1 N is
~0.3 N·m and ~2.4 rad/s of roll inside one 20 ms step. Values much above 1 N let
exploration noise alone tumble the vehicle before anything is learned.

**Degrees vs radians.** `SimResult.euler`/`omega` are degrees; `DroneState.euler`/`omega`
are radians. The metrics and plots assume the former.

**`compare()` and `scorecard()` both return strings.** They do not print. `Simulator.run`
prints the scorecard itself when `verbose=True`.

**Plot functions return a figure and never call `plt.show()`.** Showing is the caller's
decision — `viz.show()` for a window, `viz.save_plots()` for files. A `plt.show()` inside
a plotting function blocks a script and makes the figure impossible to save.

**`presets.py` exists to break a cycle.** `report.py` needs `build_controller`, and
`cli.py` needs `report`. Both import from `presets`, so neither entry point imports the
other. Do not move those definitions back into `cli.py`.

**Video costs far more than physics.** A 24 s figure-8 is ~0.8 s without video and tens
of seconds with it, all in software OpenGL. Every example takes `--no-video`.

---

## 8. Scoring

```
J = mean|p - p_ref|² + 0.01·mean|f - f_hover|² + 10·T_contact
```

Lower is better; a crash scores infinity. The RL reward is this integrand negated, so
maximizing return minimizes the score — plus a survival bonus and a command-rate
penalty, which the score itself does not include.

`EnvConfig` also exposes `w_jerk` and `w_omega` for trajectory smoothness, **both 0 by
default**. Under differential flatness the body rates are a function of jerk, so
penalising jerk is the principled way to ask for a smooth path. Useful values are around
`w_jerk = 3e-5`; expect to trade tracking accuracy for it.

---

## 9. Measured behaviour

Reference numbers on a 2-pillar-plus-crate course, so a regression is visible
(hardware: CPU, no GPU):

| task | MPC | PD | MyController | PPO |
|---|---|---|---|---|
| hover, RMSE | 0.518 m | 0.591 m | 0.591 m | 0.610 m |
| figure-8, RMSE | 0.078 m | 0.119 m | 0.119 m | 0.133 m |

`MyController` ships as the PD baseline, so it matching PD exactly is correct.

**PPO is not the best here, and that is expected.** With no wind, PD is already close to
optimal on these tasks and a residual policy has little headroom — it mostly adds
exploration noise. In a windy variant the ordering reverses sharply. Do not read this as
a broken trainer: training converges, crashes are zero, and a zero action reproduces PD
exactly by construction.

Throughput: a 24 s figure-8 with PD runs in ~0.8 s wall (roughly 30× real time, no
video); the RL env does ~2,200 steps/s per process. Video recording dominates everything
else — software OpenGL is far slower than the physics.

Outputs land in `out/<controller>_<task>/`: `flight.mp4` (1280×720 H.264),
`states.png`, `trajectory.png`, `inputs.png`.

---

## 10. Testing

`tests/test_smoke.py` covers the contract rather than the numbers: every controller
flies every task without crashing, the IMU is populated and normalised, obstacles are
present in the compiled model, and the camera renders both channels. It does not assert
tracking accuracy — that is a tuning question, not a correctness one.

Rendering needs a working OpenGL backend. On a machine that has the NVIDIA EGL vendor
file but no driver, force Mesa:

```bash
MUJOCO_GL=egl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
```
