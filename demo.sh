#!/bin/bash
# Live demo of mjc_sitl_px4 -- three controllers flown through ROS 2 SITL.
#
#   ./demo.sh show          open the recorded flights          (instant)
#   ./demo.sh pd            fly PD through ROS 2 SITL          (~60 s)
#   ./demo.sh mpc           fly MPC (sim clock at 0.2x)        (~150 s)
#   ./demo.sh ppo           fly the trained PPO policy         (~60 s)
#   ./demo.sh parity        sim and SITL share physics?        (5 s)
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
export MUJOCO_GL=${MUJOCO_GL:-egl}
run() { pkill -9 -f 'lib/mujoco_px[4]/' 2>/dev/null || true
        python3 -m mjc_sitl run --task figure8 --duration 24 --video-height 480 "$@"; }

case "${1:-show}" in
  show)
    echo "== recorded flights: PD, MPC, PPO through ROS 2 SITL =="
    for c in pd mpc ppo; do
      [ -f "demo/${c}_figure8/flight.mp4" ] && {
        echo "  demo/${c}_figure8/"; xdg-open "demo/${c}_figure8/flight.mp4" >/dev/null 2>&1 & }
    done
    sleep 1; echo "opened what exists."
    ;;
  pd)     run --controller pd ;;
  # MPC needs ~8 ms per solve; at 5 ms per physics step it cannot keep up in
  # wall-clock time, so the sim clock is slowed. The control loop is unchanged.
  mpc)    run --controller mpc --rate-scale 0.2 ;;
  ppo)    run --controller ppo --policy sim_mjc/policies/ppo_figure8.pt ;;
  all)    "$0" pd; "$0" mpc; "$0" ppo ;;
  parity) python3 -m mjc_sitl parity ;;
  *) echo "usage: ./demo.sh [show|pd|mpc|ppo|all|parity]"; exit 1 ;;
esac
