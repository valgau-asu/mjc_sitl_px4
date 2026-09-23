"""Smoke tests: every controller flies every task without crashing."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim_mjc import PDController, Simulator, make_task
from sim_mjc.controllers import LinearMPC, MyController

OBSTACLES = [{"type": "pillar", "pos": (0.0, 2.6, 1.0)},
             {"type": "box", "pos": (3.4, 2.2, 0.5)}]


@pytest.fixture(scope="module")
def sim():
    return Simulator(obstacles=OBSTACLES)


@pytest.mark.parametrize("task", ["hover", "figure8"])
@pytest.mark.parametrize("factory", [PDController, LinearMPC, MyController],
                         ids=["pd", "mpc", "mine"])
def test_flies(sim, task, factory):
    res = sim.run(factory(make_task(task)), duration=6.0, verbose=False)
    assert not res.metrics["crashed"], res.metrics["crash_reason"]
    assert np.isfinite(res.metrics["score"])
    assert len(res.t) > 0


def test_imu_is_populated(sim):
    res = sim.run(PDController(make_task("hover")), duration=2.0, verbose=False)
    assert len(res.imu) == len(res.t)
    s = res.imu[-1]
    assert np.isclose(np.linalg.norm(s.quat), 1.0, atol=1e-6)
    assert s.gyro.shape == (3,) and s.accel.shape == (3,)


def test_obstacles_are_in_the_model(sim):
    import mujoco
    names = [mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_BODY, i)
             for i in range(sim.model.nbody)]
    assert any(n and n.startswith("pillar") for n in names)
    assert any(n and n.startswith("box") for n in names)


def test_camera_renders(sim):
    from sim_mjc.sensors import Camera
    sim.reset()
    cam = Camera(sim.model, (120, 160))
    rgb, depth = cam.both(sim.data)
    cam.close()
    assert rgb.shape == (120, 160, 3)
    assert depth.shape == (120, 160) and np.isfinite(depth).any()
