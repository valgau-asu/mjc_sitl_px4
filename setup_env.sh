# Source this before running the SITL nodes. Order matters: px4_ws first (supplies
# px4_msgs), this workspace last so its mujoco_px4 shadows the older copy
# installed in px4_ws.
#
# Paths are resolved relative to this file so the workspace can be cloned
# anywhere. Override the px4_msgs location with:  export PX4_WS=/path/to/px4_ws
_SITL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PX4_WS="${PX4_WS:-$HOME/px4_ws}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"

source "$ROS_SETUP"
if [ -f "$PX4_WS/install/setup.bash" ]; then
    source "$PX4_WS/install/setup.bash"
else
    echo "warning: px4_msgs not found at $PX4_WS/install -- set PX4_WS" >&2
fi
[ -f "$_SITL_DIR/install/setup.bash" ] && source "$_SITL_DIR/install/setup.bash"
