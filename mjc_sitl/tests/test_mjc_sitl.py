"""Tests for the deployment path."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from mjc_sitl import scenegen                                   # noqa: E402
from mjc_sitl.bridge import (Bridge, _match_allocations,        # noqa: E402
                               _model_allocation, rotor_permutation)
from mjc_sitl.contract import DeployContract, Tolerances        # noqa: E402
from mjc_sitl.vehicle import DART, SIM_DEFAULT, Rotor, VehicleSpec  # noqa: E402

ADAPTER = ROOT / "mjc_sitl" / "offboard_lib" / "adapter.py"


@pytest.fixture(scope="module")
def contract():
    return DeployContract(controller_kind="pd", task="hover", duration=4.0)


# ---------------------------------------------------------------- vehicle
def test_spec_matches_compiled_model(contract):
    """A VehicleSpec must describe the MJCF it claims to describe."""
    import mujoco
    model = scenegen.shared_model(contract)
    drone = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drone")
    spec = contract.spec
    assert model.body_mass[drone] == pytest.approx(spec.mass)
    assert model.body_inertia[drone] == pytest.approx([spec.Ixx, spec.Iyy, spec.Izz])
    assert model.actuator_ctrlrange[0, 1] == pytest.approx(spec.max_thrust)
    assert model.opt.timestep == pytest.approx(spec.timestep)


def test_spec_allocation_matches_model(contract):
    """The analytic allocation must equal what MuJoCo actually applies."""
    model = scenegen.shared_model(contract)
    assert np.allclose(_model_allocation(model), contract.spec.allocation(), atol=1e-9)


def test_twr_is_sane():
    for spec in (SIM_DEFAULT, DART):
        assert spec.twr > 1.5, f"{spec.name} cannot lift itself with margin"


# ---------------------------------------------------------------- bridge
def test_permutation_is_unique_and_correct():
    """Exactly one ordering maps sim_mjc onto dart -- 1 of 24, not a guess."""
    perm = rotor_permutation(SIM_DEFAULT, DART)
    assert list(perm) == [1, 3, 0, 2]
    for j, i in enumerate(perm):
        assert (SIM_DEFAULT.rotors[i].sx, SIM_DEFAULT.rotors[i].sy) == \
               (DART.rotors[j].sx, DART.rotors[j].sy)
        assert SIM_DEFAULT.rotors[i].spin == DART.rotors[j].spin


def test_permutation_matches_compiled_models():
    """The spec-derived mapping must equal the one read out of the MJCFs."""
    import mujoco
    sim = scenegen.shared_model(DeployContract(scene="generated"))
    dart_path = scenegen.DART_SCENE
    if not dart_path.exists():
        pytest.skip("legacy dart scene not present")
    dart = mujoco.MjModel.from_xml_path(str(dart_path))
    got = _match_allocations(_model_allocation(sim), _model_allocation(dart))
    assert got is not None, "no consistent mapping between the compiled models"
    assert list(got) == list(rotor_permutation(SIM_DEFAULT, DART))


def test_identity_bridge_is_identity():
    b = Bridge(SIM_DEFAULT, SIM_DEFAULT)
    assert b.is_identity and b.same_airframe
    u = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.allclose(b.thrusts(u), u)


def test_bridge_clips_to_target():
    b = Bridge(SIM_DEFAULT, SIM_DEFAULT)
    assert b.thrusts([-5, 0, 99, 3]).tolist() == [0.0, 0.0, SIM_DEFAULT.max_thrust, 3.0]


def test_ambiguous_layout_is_refused():
    """A symmetric airframe cannot be mapped from geometry -- refuse, don't guess."""
    sym = VehicleSpec(
        name="symmetric", mass=1.0, arm=0.1, max_thrust=5.0, k_m=0.01,
        Ixx=0.01, Iyy=0.01, Izz=0.02, body_size=(0.04, 0.04, 0.02),
        rotor_radius=0.05, timestep=0.005,
        rotors=tuple(Rotor(f"r{i}", +1, +1, +1) for i in range(4)))
    with pytest.raises(ValueError, match="ambiguous"):
        rotor_permutation(sym, sym)


def test_spin_disagreement_is_diagnosed():
    """Same positions, flipped spin: must name the yaw-sign fault specifically."""
    flipped = DART.with_(rotors=tuple(
        Rotor(r.name, r.sx, r.sy, -r.spin) for r in DART.rotors))
    with pytest.raises(ValueError, match="spin direction"):
        rotor_permutation(SIM_DEFAULT, flipped)


# ---------------------------------------------------------------- contract
def test_contract_roundtrip(tmp_path):
    c = DeployContract(controller_kind="mpc", task="figure8", control_hz=25,
                       tolerances=Tolerances(max_pos_rmse=0.42))
    p = c.save(tmp_path / "c.json")
    back = DeployContract.load(p)
    assert back == c and back.tolerances.max_pos_rmse == 0.42


def test_control_rate_must_divide_physics():
    with pytest.raises(ValueError, match="must divide"):
        DeployContract(control_hz=37)


def test_ppo_without_policy_is_refused():
    with pytest.raises(ValueError, match="policy_path"):
        DeployContract(controller_kind="ppo")


# ---------------------------------------------------------------- parity
def test_shared_physics(contract):
    rep = scenegen.verify_shared_physics(contract)
    assert rep["ok"], rep["problems"]
    assert rep["checks"], "no checks ran"


# ---------------------------------------------------------------- validators
def test_coding_accepts_the_default_adapter(contract):
    from mjc_sitl.validators import coding
    assert coding.validate(ADAPTER, contract).passed


def test_coding_rejects_ros_import(tmp_path, contract):
    from mjc_sitl.validators import coding
    bad = tmp_path / "a.py"
    bad.write_text("import rclpy\n\ndef build(contract):\n    return None\n")
    r = coding.validate(bad, contract)
    assert not r.passed and "ros_import" in [f.code for f in r.errors]


def test_coding_rejects_missing_build(tmp_path, contract):
    from mjc_sitl.validators import coding
    bad = tmp_path / "a.py"
    bad.write_text("x = 1\n")
    r = coding.validate(bad, contract)
    assert "no_build" in [f.code for f in r.errors]


def test_coding_rejects_nan_command(tmp_path, contract):
    from mjc_sitl.validators import coding
    bad = tmp_path / "a.py"
    bad.write_text(
        "import numpy as np\n"
        "class C:\n"
        "    def control(self, t, s):\n"
        "        return np.array([np.nan, 0.0, 0.0, 0.0])\n"
        "def build(contract):\n    return C()\n")
    r = coding.validate(bad, contract)
    assert "non_finite" in [f.code for f in r.errors]


def test_coding_rejects_wrong_shape(tmp_path, contract):
    from mjc_sitl.validators import coding
    bad = tmp_path / "a.py"
    bad.write_text(
        "import numpy as np\n"
        "class C:\n"
        "    def control(self, t, s):\n        return np.zeros(3)\n"
        "def build(contract):\n    return C()\n")
    assert "bad_shape" in [f.code for f in coding.validate(bad, contract).errors]


def test_findings_carry_hints(tmp_path, contract):
    """Every error must say what to do about it."""
    from mjc_sitl.validators import coding
    bad = tmp_path / "a.py"
    bad.write_text("import rclpy\n")
    for f in coding.validate(bad, contract).errors:
        assert f.hint, f"finding {f.code} has no actionable hint"


# ---------------------------------------------------------------- dynamics
def test_fast_dynamics_matches_sim(contract):
    """Shared model + shared controller: SITL must track sim closely."""
    from mjc_sitl.validators import dynamics
    r = dynamics.validate(contract, ADAPTER, "/tmp/fd_test", tier="fast")
    assert r.passed, r.report()
    assert r.metrics["pos_rmse"] < contract.tolerances.max_pos_rmse


def test_safety_passes_for_pd(contract):
    from mjc_sitl.validators import dynamics, safety
    run = dynamics.run_fast_sitl(contract, ADAPTER)
    assert safety.validate(run, contract).passed


def test_safety_catches_saturation(contract):
    """A command pinned at the rail must be refused, not merely noted."""
    from mjc_sitl.validators import safety
    n = 500
    run = {"thrusts": np.full((n, 4), contract.spec.max_thrust),
           "pos": np.tile([0, 0, 1.0], (n, 1)), "vel": np.zeros((n, 3)),
           "crashed": False, "contacts": 0}
    r = safety.validate(run, contract)
    assert not r.passed and "saturated" in [f.code for f in r.errors]


# ---------------------------------------------------------------- misc
# ---------------------------------------------------------------- slow
@pytest.mark.slow
def test_full_sitl_matches_sim(contract):
    """The real ROS 2 stack, end to end. Needs a sourced ROS 2 environment."""
    if shutil.which("ros2") is None:
        pytest.skip("ROS 2 not on PATH")
    from mjc_sitl.validators import dynamics
    r = dynamics.validate(contract, ADAPTER, "/tmp/fd_test_full", tier="full")
    assert r.passed, r.report()

