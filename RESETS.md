# YuMi Reset Overview

This module handles automatic and manual reset procedures for the YuMi robot.

## Reset Mechanism

YuMi requires two logic switches to transition from manual to automatic mode and to re-enable the motors after a failure. This involves sending two I/O signals using the Jacobi interface. These actions are also triggered automatically when reinitializing the YuMi interface.

## Automatic Reset Behavior

When a failure occurs, the system prints the error and automatically tries to reset. This logic is called from `dependencies/yumi_jacobi/yumi.py`, inside the `_on_fail` function. You can change the behavior for different errors there.

This function calls `_on_interface_fail` in the `YuMiRobotEnv` class (`cable_routing/env/robot/yumi.py`). To customize the reset behavior, edit `_on_interface_reset_request` in the same class. It currently adds a delay and retries, but you can replace this with custom rescue motions if needed.

## Manual Reset Support

Manual reset is also supported via a ROS service:

```python
rospy.Service("/manual_reset_yumi", Trigger, self._handle_manual_reset)
```

This allows manual intervention when needed, particularly useful during debugging or development.
