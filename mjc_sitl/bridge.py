"""Frame, unit and rotor-order conversion between sim_mjc and the ROS 2 node.

qpos[0:3] world position, qpos[3:7] quaternion (w,x,y,z), qvel[0:3] world
linear velocity, qvel[3:6] body angular velocity. sim_mjc positions are
zone-local; raw qpos is world.
"""
import itertools
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .vehicle import VehicleSpec


def rotor_permutation(src: VehicleSpec, dst: VehicleSpec) -> np.ndarray:
    """Indices `perm` such that dst_thrusts = src_thrusts[perm].
    
    Matched by body-frame position and checked against spin sign. Raises unless
    exactly one ordering fits.
    """
    if len(src.rotors) != len(dst.rotors):
        raise ValueError(f"rotor count differs: {src.name} has "
                         f"{len(src.rotors)}, {dst.name} has {len(dst.rotors)}")
    n = len(src.rotors)
    src_key = [(r.sx, r.sy, r.spin) for r in src.rotors]
    dst_key = [(r.sx, r.sy, r.spin) for r in dst.rotors]

    hits = [p for p in itertools.permutations(range(n))
            if [src_key[i] for i in p] == dst_key]
    if not hits:
        pos_only = [p for p in itertools.permutations(range(n))
                    if [src_key[i][:2] for i in p] == [k[:2] for k in dst_key]]
        if pos_only:
            raise ValueError(
                f"{src.name} and {dst.name} place their rotors identically but "
                f"disagree on spin direction: {src.name}={[k[2] for k in src_key]} "
                f"vs {dst.name}={[k[2] for k in dst_key]} once aligned. One of the "
                f"two MJCFs has its `gear` yaw sign wrong; that inverts yaw only, "
                f"so it will not look like a rotor-order bug.")
        raise ValueError(f"no rotor ordering maps {src.name} onto {dst.name}: "
                         f"{src_key} vs {dst_key}")
    if len(hits) > 1:
        raise ValueError(
            f"rotor mapping {src.name} -> {dst.name} is ambiguous "
            f"({len(hits)} orderings fit: {hits}). A symmetric layout cannot be "
            f"disambiguated from geometry alone; name the mapping explicitly.")
    return np.array(hits[0], dtype=int)


@dataclass
class Bridge:
    """Converts thrust vectors and positions between two airframes."""
    src: VehicleSpec
    dst: VehicleSpec
    zone_offset: np.ndarray = None

    def __post_init__(self):
        self.perm = rotor_permutation(self.src, self.dst)
        self.zone_offset = (np.zeros(3) if self.zone_offset is None
                            else np.asarray(self.zone_offset, float))

    # -- properties ---------------------------------------------------------
    @property
    def is_identity(self) -> bool:
        return bool(np.array_equal(self.perm, np.arange(len(self.perm))))

    @property
    def same_airframe(self) -> bool:
        """True when both sides are the same physical vehicle.
        
        Rotor order does not count: two builds can wire motors differently and still
        be one airframe.
        """
        a, b = self.src, self.dst
        return (np.isclose(a.mass, b.mass)
                and np.isclose(a.arm, b.arm)
                and np.isclose(a.max_thrust, b.max_thrust)
                and np.isclose(a.k_m, b.k_m)
                and np.allclose([a.Ixx, a.Iyy, a.Izz], [b.Ixx, b.Iyy, b.Izz])
                and np.isclose(a.timestep, b.timestep))

    # -- transforms ---------------------------------------------------------
    def thrusts(self, u: Sequence[float]) -> np.ndarray:
        """Reorder into dst actuator order and clip to dst limits."""
        u = np.asarray(u, float).ravel()
        if u.shape != (len(self.perm),):
            raise ValueError(f"expected {len(self.perm)} thrusts, got {u.shape}")
        return np.clip(u[self.perm], 0.0, self.dst.max_thrust)

    def to_local(self, world_pos) -> np.ndarray:
        """World position -> zone-local."""
        return np.asarray(world_pos, float) - self.zone_offset

    def to_world(self, local_pos) -> np.ndarray:
        return np.asarray(local_pos, float) + self.zone_offset

    # -- verification -------------------------------------------------------
    def verify_against_models(self, src_model, dst_model, atol=1e-9) -> dict:
        """Re-derive the permutation from two compiled MjModels and assert it matches."""
        A_src = _model_allocation(src_model)
        A_dst = _model_allocation(dst_model)
        got = _match_allocations(A_src, A_dst)
        if got is None:
            raise AssertionError(
                f"the compiled models admit no consistent rotor mapping.\n"
                f"  {self.src.name} allocation:\n{A_src}\n"
                f"  {self.dst.name} allocation:\n{A_dst}")
        if not np.array_equal(got, self.perm):
            raise AssertionError(
                f"rotor mapping disagrees with the compiled models: spec says "
                f"{list(self.perm)}, models say {list(got)}. A VehicleSpec has "
                f"drifted from its MJCF.")
        return {"permutation": list(map(int, self.perm)),
                "src_allocation": A_src, "dst_allocation": A_dst,
                "identity": self.is_identity,
                "same_airframe": self.same_airframe}

    def report(self) -> str:
        L = [f"rotor bridge  {self.src.name} -> {self.dst.name}"]
        for j, i in enumerate(self.perm):
            L.append(f"  {self.dst.rotors[j].name:<8} <- {self.src.rotors[i].name:<8}"
                     f"  (offset {self.src.rotors[i].sx:+d},{self.src.rotors[i].sy:+d}"
                     f"  spin {self.src.rotors[i].spin:+d})")
        L.append(f"  permutation      {list(map(int, self.perm))}"
                 f"{'  (identity)' if self.is_identity else ''}")
        L.append(f"  same airframe    {self.same_airframe}")
        if not self.same_airframe:
            L.append("  NOTE: the airframes differ physically; a dynamics "
                     "comparison across this bridge measures that gap too. "
                     "values.")
        if np.any(self.zone_offset):
            L.append(f"  zone offset      {self.zone_offset}")
        return "\n".join(L)


# --------------------------------------------------------------------------
# Deriving allocation straight from a compiled model.
# --------------------------------------------------------------------------
def _model_allocation(model) -> np.ndarray:
    """(4, nu) [T, tau_x, tau_y, tau_z] per unit control, at identity attitude."""
    import mujoco
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]        # identity: body frame == world
    cols = []
    for i in range(model.nu):
        data.ctrl[:] = 0.0
        data.ctrl[i] = 1.0
        mujoco.mj_forward(model, data)
        q = data.qfrc_actuator.copy()
        cols.append([q[2], q[3], q[4], q[5]])     # Fz, tau_x, tau_y, tau_z
    return np.array(cols).T


def _match_allocations(A_src: np.ndarray, A_dst: np.ndarray) -> Optional[np.ndarray]:
    """Permutation p with A_src[:, p] ~ A_dst up to per-axis scale, else None."""
    def structure(A):
        tau = A[1:4]
        scale = np.abs(tau[:2]).max()
        return np.sign(np.round(tau / scale, 6)) if scale > 0 else np.sign(tau)

    S_src, S_dst = structure(A_src), structure(A_dst)
    hits = [p for p in itertools.permutations(range(A_src.shape[1]))
            if np.array_equal(S_src[:, list(p)], S_dst)]
    return np.array(hits[0], int) if len(hits) == 1 else None
