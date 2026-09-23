# Attribution

This directory is vendored from **<https://github.com/YogeshMaan/mujoco_px4>**
(branch `master`, base commit `c4a1f81`), © Yogesh Maan, plus local modifications.

It is the original MuJoCo SITL stack this project grew out of: a MuJoCo stepper
(`quad_sim`), a Lee SE(3) geometric controller (`geometeric_controller`), and an
Informed-RRT* + minimum-snap planner (`trajectory_planner_rrt`).

The upstream repository declares no explicit licence. It is included here with
attribution for reference and teaching. If you intend to reuse it, please contact the
author.

Local modifications: scene path resolved via `ament_index` instead of an absolute
path; `setup.py` installs the model files; airframe constants read from
`vehicle_params.py` rather than being hardcoded.
