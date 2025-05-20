from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
import traceback
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored, cprint
from autolab_core import RigidTransform, Point, CameraIntrinsics
import datetime
import time


def normalize(vec):
    return vec / np.linalg.norm(vec)


# Attempts to execute a robot motion function (e.g., plan_and_execute_linear_waypoints)
# with fallback mechanisms. Solving Jacobi's issues
# This utility is designed to ensure motion commands are executed reliably by:
#   1. Executing the function with timeout protection.
#   2. Retrying with simplified (constant orientation) waypoints if initial planning fails.
#   3. As a last resort, directly commanding the robot to the final pose using set_ee_pose
#      (if a fallback_func is provided).
# This structure helps recover from transient planning errors, avoids full task failures,
# and allows more graceful degradation of motion precision in tightly constrained environments.


def run_with_fallback(
    func, timeout=15, *args, checks=True, fallback_func=None, **kwargs
):
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
    import traceback
    from termcolor import cprint

    def _try_exec(f, *args, **kwargs):
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(f, *args, **kwargs)
            try:
                result = future.result(timeout=timeout)
                if checks and result is True:
                    cprint(
                        f"[Fail] Function `{f.__name__}` returned True (planning failed)",
                        "red",
                    )
                    return "planning_failed"
                return result
            except FuturesTimeout:
                cprint(
                    f"[Timeout] `{f.__name__}` timed out after {timeout} seconds.",
                    "yellow",
                )
                return "timeout"
            except Exception as e:
                cprint(f"[Error] `{f.__name__}` raised: {e}", "yellow")
                traceback.print_exc()
                return "exception"

    result = _try_exec(func, *args, **kwargs)

    if result in ("planning_failed", "timeout", "exception"):
        if (
            func.__name__ == "plan_and_execute_linear_waypoints"
            and "waypoints" in kwargs
        ):
            cprint("[Retry] Motion failed — retrying with fallback rotation", "cyan")
            original_waypoints = kwargs["waypoints"]

            try:
                if isinstance(original_waypoints, list):
                    rot = original_waypoints[0].rotation
                    fallback = [
                        RigidTransform(translation=wp.translation, rotation=rot)
                        for wp in original_waypoints
                    ]
                elif isinstance(original_waypoints, tuple):
                    fallback = []
                    for arm_waypoints in original_waypoints:
                        if arm_waypoints:
                            rot = arm_waypoints[0].rotation
                            fallback.append(
                                [
                                    RigidTransform(
                                        translation=wp.translation, rotation=rot
                                    )
                                    for wp in arm_waypoints
                                ]
                            )
                        else:
                            fallback.append([])
                    fallback = tuple(fallback)
                else:
                    cprint("[Retry Abort] Unknown waypoint format", "red")
                    return None

                new_kwargs = dict(kwargs)
                new_kwargs["waypoints"] = fallback
                fallback_result = _try_exec(func, *args, **new_kwargs)

                if fallback_result not in ("planning_failed", "timeout", "exception"):
                    return fallback_result

                cprint(f"[Retry] Fallback result: {fallback_result}", "cyan")

            except Exception as e:
                cprint(f"[Retry Failed] Could not build fallback waypoints: {e}", "red")
                traceback.print_exc()

        # FINAL fallback to set_ee_pose
        if fallback_func:
            arm = args[0] if args else kwargs.get("arm", None)
            waypoints = kwargs.get("waypoints")

            if not waypoints:
                cprint("[Fallback Abort] No waypoints provided for fallback", "red")
                return None

            cprint("[Fallback] Falling back to direct pose execution", "magenta")

            try:
                if isinstance(waypoints, list):
                    final_pose = waypoints[-1]
                    if arm == "left":
                        result = fallback_func(left_pose=final_pose, right_pose=None)
                    elif arm == "right":
                        result = fallback_func(left_pose=None, right_pose=final_pose)
                    else:
                        cprint("[Fallback Abort] Unknown arm", "red")
                        return None
                elif isinstance(waypoints, tuple) and len(waypoints) == 2:
                    result = fallback_func(
                        left_pose=waypoints[0][-1] if waypoints[0] else None,
                        right_pose=waypoints[1][-1] if waypoints[1] else None,
                    )
                else:
                    cprint("[Fallback Abort] Unknown waypoint format", "red")
                    return None

                return result

            except Exception as e:
                cprint(f"[Fallback Error] {e}", "red")
                traceback.print_exc()
                exit(1)

    return result


def need_regrasp(curr_clip, next_clip, prev_clip, arm):
    curr_pos = np.array([curr_clip["x"], curr_clip["y"]])
    next_pos = np.array([next_clip["x"], next_clip["y"]])

    direction = next_clip["x"] - curr_clip["x"]
    if np.linalg.norm(next_pos - curr_pos) <= 100:
        return False

    if arm == "right":
        return direction > 0
    if arm == "left":
        return direction < 0

    raise ValueError(f"Invalid arm: {arm}")


def calculate_sequence(curr_clip, prev_clip, next_clip):
    """
    determines how to wrap around a give clip
    """

    curr_x, curr_y = curr_clip["x"], curr_clip["y"]
    prev_x, prev_y = prev_clip["x"], prev_clip["y"]
    next_x, next_y = next_clip["x"], next_clip["y"]

    num2dir = {0: "up", 1: "right", 2: "down", 3: "left"}
    dir2num = {val: key for key, val in num2dir.items()}
    clip_vecs = np.array([[0, 1, 0], [1, 0, 0], [0, -1, 0], [-1, 0, 0]])
    prev2curr = normalize(np.array([curr_x - prev_x, -(curr_y - prev_y), 0]))
    curr2prev = -prev2curr
    curr2next = normalize(np.array([next_x - curr_x, -(next_y - curr_y), 0]))
    clip_vec = clip_vecs[(curr_clip["orientation"] // 90 + 1) % 4]
    is_clockwise = np.cross(prev2curr, curr2next)[-1] > 0

    net_vector = curr2prev + curr2next
    if abs(net_vector[0]) > abs(net_vector[1]):
        if net_vector[0] > 0:
            middle_node = dir2num["left"]
        else:
            middle_node = dir2num["right"]
    else:
        if net_vector[1] > 0:
            middle_node = dir2num["down"]
        else:
            middle_node = dir2num["up"]

    if is_clockwise:
        sequence = [
            num2dir[(middle_node + 1) % 4],
            num2dir[middle_node],
            num2dir[(middle_node - 1) % 4],
        ]
    else:
        sequence = [
            num2dir[(middle_node - 1) % 4],
            num2dir[middle_node],
            num2dir[(middle_node + 1) % 4],
        ]

    return sequence, -1 if is_clockwise else 1
