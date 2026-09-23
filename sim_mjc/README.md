# sim_mjc

A modular MuJoCo quadrotor simulation environment for control research.

Four controllers — cascaded PD, linear MPC, a trained PPO policy, and your own — each
flying two standard tasks: **hover** (setpoint tracking) and **figure-8**. Obstacles,
an IMU and an onboard camera are part of the model. There is no wind: given a
controller and an initial condition, a run is deterministic.

## Outputs

Every example writes what it produced into `out/<controller>_<task>/` — a 1280×720
chase-camera `flight.mp4` and three figures (`states`, `trajectory`, `inputs`):

```bash
python3 examples/run_pd.py                 # both tasks, plots + video
python3 examples/run_mpc.py --task figure8
python3 examples/run_ppo.py --no-video     # skip rendering, much faster
python3 examples/run_pd.py --show          # display the figures as well

python -m sim_mjc.cli --controller mpc --task figure8 --plot --video
python -m sim_mjc.cli --controller pd --task hover --plot --out /tmp/runs
```

Rendering needs an OpenGL backend. With a display attached it works as-is; headless,
set `MUJOCO_GL=egl` (or `osmesa`).

## Install

```bash
pip install mujoco numpy scipy            # core
pip install cvxpy                         # MPC
pip install torch gymnasium               # RL
pip install matplotlib mediapy            # plots and video
```

Then either work from this directory or `pip install -e .`.

## Run

```bash
python -m sim_mjc.cli --controller pd   --task hover
python -m sim_mjc.cli --controller mpc  --task figure8 --plot
python -m sim_mjc.cli --controller mine --task figure8
python -m sim_mjc.cli --compare --task figure8

python -m sim_mjc.rl.train --task figure8 --steps 300000
python -m sim_mjc.cli --controller ppo --task figure8 --policy policies/ppo_figure8.pt
```

`examples/` holds the same runs as plain scripts.

## Use as a library

```python
from sim_mjc import Simulator, PDController, setpoint

sim = Simulator(obstacles=[{"type": "pillar", "pos": (2, 0, 1)}])
res = sim.run(PDController(setpoint([1, 1, 1.5])), duration=10.0)

print(res.metrics["rmse"], res.metrics["score"])
print(res.imu[-1].gyro, res.imu[-1].accel)
```

The model is compiled once per `Simulator`, so repeated `run()` calls are cheap.

## Writing a controller

Anything with a `control(t, state) -> 4 rotor thrusts [N]` method works. Start from
`sim_mjc/controllers/my_controller.py`.

`state` carries the ground truth used for scoring — `pos`, `vel`, `quat`, `euler`,
`omega` — and `state.imu`, which is what the onboard sensors actually measure:

| field | meaning |
|---|---|
| `imu.gyro` | body angular rate [rad/s] |
| `imu.accel` | proper acceleration, body frame [m/s²] |
| `imu.quat` | orientation `(w, x, y, z)` |
| `imu.pos` | position from the site frame [m] |
| `imu.linvel` | linear velocity, world frame [m/s] |

Use `state.imu` if the controller should be honest about what a real vehicle can
observe; use the ground-truth fields if you do not care.

## Layout

```
sim_mjc/
├── config.py          vehicle, arena and obstacle definitions
├── model/             MJCF generation (xml.py) and scene assembly (scene.py)
├── dynamics.py        mixer, allocation, hover linearization
├── sensors.py         DroneState, IMU, Camera
├── simulator.py       the flight loop
├── trajectories.py    references, and the two standard tasks
├── metrics.py         scoring
├── viz.py             plots and video
├── cli.py             one entry point for every controller and task
├── controllers/       base, pd, mpc, ppo, my_controller
└── rl/                env.py (Gymnasium), ppo.py (trainer), train.py (CLI)
```

## The model

* **Vehicle** 1.25 kg, 0.15 m arm, 7 N per rotor (thrust-to-weight 2.3), rotor order
  `[FR, BR, BL, FL]`.
* **Rates** physics 200 Hz, control 50 Hz with a zero-order hold.
* **Arena** a scale model of the ASU Drone Studio: six 11.28 m zones. Pick one with
  `zone="D"`; the controller always sees the middle of its zone as the origin.
* **Obstacles** `pillar` (static cylinder), `box` (free-floating 1 m cube, 1.5 kg — the
  drone can knock it over), `gate` (1.4 m square opening). Positions are given in the
  zone-local frame. Nothing tells the controller they are there.
* **Sensors** gyro, accelerometer, orientation, position and linear velocity at an
  `imu` site.
* **Cameras** `onboard` (90° FOV, RGB and metric depth), plus `chase` and `track` for
  video.

## Scoring

```
J = mean|p - p_ref|² + 0.01·mean|f - f_hover|² + 10·T_contact
```

Lower is better; a crash scores infinity. `compare(*runs)` prints a table.

## Tests

```bash
pytest tests/
```
