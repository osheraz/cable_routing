import numpy as np
from autolab_core import RigidTransform, Point
from typing import List, Tuple, Union, Literal

from yumi_jacobi.yumi import YuMiArm
import asyncio
from jacobi import Frame, CartesianWaypoint, LinearMotion, Motion, Studio, Trajectory
from jacobi import Planner
from jacobi.robots import ABBYuMiIRB14000 as Yumi
from jacobi.drivers import ABBDriver
from yumi_jacobi.tcp import *
from copy import copy
import os
import time
import traceback

import threading


class Interface:
    GRIP_DOWN_R = np.diag([1, -1, -1])
    GRIP_DOWN_L = np.diag([-1, 1, -1])
    GRIP_UP_R = np.diag([1, 1, 1])
    GRIP_SIDEWAYS = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

    def __init__(self, *args, on_fail=None, **kwargs):
        
        self._async_interface = AsyncInterface(*args, **kwargs, on_fail=on_fail)
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._start_loop, daemon=True)
        self._thread.start()
        self.yumi = self._async_interface.yumi
        self.planner = self._async_interface.planner
        self.driver_left = self._async_interface.driver_left
        self.driver_right = self._async_interface.driver_right
            
    def _start_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_coroutine(self, coro):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()
    
    def is_ready(self):
        try:
            # Try accessing a basic property to check connection
            _ = self.driver_left.current_joint_position
            return True
        except:
            return False 

    def calibrate_grippers(self):
        """
        Calibrate both grippers
        """
        return self._run_coroutine(self._async_interface.calibrate_grippers())

    def close_grippers(self, side: Literal["both", "left", "right"] = "both"):
        """
        Close grippers on the specified side ['both','left','right']
        default = 'both'
        """
        return self._run_coroutine(self._async_interface.close_grippers(side))

    def open_grippers(self, side: Literal["both", "left", "right"] = "both"):
        """
        Open grippers on the specified side ['both','left','right']
        default = 'both'
        """
        return self._run_coroutine(self._async_interface.open_grippers(side))

    def grippers_move_to(self, left_dist: int = 0, right_dist: int = 0):
        """
        Jog grippers to the specified width [0, 25] (mm)
        Default arguments are 0 mm open grippers
        """
        return self._run_coroutine(
            self._async_interface.grippers_move_to(left_dist, right_dist)
        )

    def home(self):
        """
        Move both arms to the home position
        """
        return self._run_coroutine(self._async_interface.home())

    def home2(self):
        """
        Move both arms to the home position
        """
        return self._run_coroutine(self._async_interface.home2())

    def move_to(
        self, left_goal: List = None, right_goal: List = None, ignore_collisions=False
    ):
        """
        Move both arms to the specified joint angles (radians)
        """
        return self._run_coroutine(
            self._async_interface.move_to(
                left_goal, right_goal, ignore_collisions=ignore_collisions
            )
        )

    def go_delta(self, left_delta=[0, 0, 0], right_delta=[0, 0, 0]):
        """
        Move both arms by the specified cartesian delta [x,y,z], meant for small motions
        """
        return self._run_coroutine(
            self._async_interface.go_delta(left_delta, right_delta)
        )

    def go_cartesian_waypoints(
        self, l_targets: List[RigidTransform] = [], r_targets: List[RigidTransform] = []
    ):
        """
        Move both arms to waypoint [RigidTransform] or along specified waypoints (list of RigidTransforms)
        Currently Jacobi supports up to 3 waypoints
        """
        return self._run_coroutine(
            self._async_interface.go_cartesian_waypoints(l_targets, r_targets)
        )

    def go_linear_single(
        self, l_target: RigidTransform = None, r_target: RigidTransform = None
    ):
        """
        Move both arms linearly to the specified waypoint
        """
        return self._run_coroutine(
            self._async_interface.go_linear_single(l_target, r_target)
        )

    def run_trajectory(
        self, l_trajectory: Trajectory = None, r_trajectory: Trajectory = None
    ):
        """
        Run the planned trajectories for both arms
        """
        return self._run_coroutine(
            self._async_interface.run_trajectory(l_trajectory, r_trajectory)
        )

    def run_trajectory_no_async(
        self, l_trajectory: Trajectory = None, r_trajectory: Trajectory = None
    ):
        """
        Run the planned trajectories for both arms
        """
        return self._async_interface.run_trajectory_no_async(l_trajectory, r_trajectory)

    def run_trajectories(self, trajectories: List[Trajectory]):
        """
        Run the planned trajectories for both arms
        """
        return self._run_coroutine(self._async_interface.run_trajectories(trajectories))

    def blend_into(self, trajectory_l: Motion, trajectory_r: Motion, duration=1.0):
        """
        Blend into the planned trajectories for both arms
        """
        return self._run_coroutine(
            self._async_interface.blend_into(
                trajectory_l, trajectory_r, duration=duration
            )
        )

    def plan_cartesian_waypoints(
        self,
        l_targets: List[RigidTransform],
        r_targets: List[RigidTransform],
        start_from_current_cfg=True,
    ) -> List[Trajectory]:
        """
        Plan cartesian waypoints for both arms
        """
        return self._async_interface.plan_cartesian_waypoints(
            l_targets, r_targets, start_from_current_cfg=start_from_current_cfg
        )

    def plan_linear_waypoints(
        self,
        l_targets: List[RigidTransform] = [],
        r_targets: List[RigidTransform] = [],
        start_from_current_cfg=True,
    ) -> List[Trajectory]:
        """
        Plan linear motion for both arms
        """
        return self._async_interface.plan_linear_waypoints(
            l_targets, r_targets, start_from_current_cfg=start_from_current_cfg
        )

    def get_FK(self, arm: Literal["left", "right"]) -> RigidTransform:
        """
        Get the forward kinematics of the specified arm, returns a RigidTransform
        """
        return self._async_interface.get_FK(arm)

    def get_joint_positions(self, arm: Literal["left", "right"]) -> List:
        """
        Get the joint positions of the specified arm, returns a list of joint angles (radians)
        """
        return self._async_interface.get_joint_positions(arm)

    def RT2Frame(self, transform: RigidTransform) -> Frame:
        """
        Convert RigidTransform (autolab_core) to Frame (Jacobi object)
        """
        return self._async_interface.RT2Frame(transform)

    def Frame2RT(self, frame: Frame) -> RigidTransform:
        """
        Convert Frame (Jacobi object) to RigidTransform (autolab_core)
        """
        return self._async_interface.Frame2RT(frame)

    def listRT2Motion(
        self, robot, start: List, wp_list: List[RigidTransform]
    ) -> Motion:
        """
        Convert list of RigidTransform to Motion (Jacobi object)
        """
        return self._async_interface.listRT2Motion(robot, start, wp_list)

    def listRT2LinearMotion(
        self, robot, start: Union[List, RigidTransform], goal: RigidTransform
    ) -> Motion:
        """
        Convert list of RigidTransform to LinearMotion (Jacobi object)
        """
        return self._async_interface.listRT2LinearMotion(robot, start, goal)

    def RT2CW(self, transform: RigidTransform) -> CartesianWaypoint:
        """
        Convert RigidTransform (autolab_core) to CartesianWaypoint (Jacobi object)
        """
        return self._async_interface.RT2CW(transform)

    def plan(self, motion: List[Motion]) -> List[Trajectory]:
        """
        Plan a list of Motions
        """
        return self.planner.plan(motion)

    def plan(self, motion: Motion) -> Trajectory:
        """
        Plan a single Motion
        """
        return self.planner.plan(motion)


class AsyncInterface:
    # orientation where the gripper is facing downwards
    GRIP_DOWN_R = np.diag([1, -1, -1])
    GRIP_DOWN_L = np.diag([-1, 1, -1])
    GRIP_UP_R = np.diag([1, 1, 1])
    GRIP_SIDEWAYS = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    L_ARMS_CLEAR_STATE = np.array(
        [
            -0.5810662,
            -1.34913424,
            0.73567095,
            0.23,
            1.56402364,
            1.25265177,
            2.9580,
        ]
    )
    R_ARMS_CLEAR_STATE = np.array(
        [
            0.64224786,
            -1.34920282,
            -0.74859683,
            0.18,
            -1.64836569,
            1.20916355,
            -2.83024169,
        ]
    )
    L_HOME_STATE = np.array(
        [
            -0.5810662,
            -1.34913424,
            0.73567095,
            0.55716616,
            1.56402364,
            1.25265177,
            2.84548536,
        ]
    )
    R_HOME_STATE = np.array(
        [
            0.64224786,
            -1.34920282,
            -0.82859683,
            0.55531042,
            -1.84836569,
            1.20916355,
            -2.83024169,
        ]
    )

    def __init__(
        self,
        speed=0.14,
        l_tcp=ABB_WHITE.as_frames("l_tcp_frame", "l_tip_frame"),
        r_tcp=ABB_WHITE.as_frames("r_tcp_frame", "r_tip_frame"),
        file=None,
        on_fail=None,
    ):

        self.speed = speed

        self.l_tcp = l_tcp
        self.r_tcp = r_tcp

        if file is not None:
            if os.path.exists(file):
                self.planner = Planner.load_from_project_file(file)
                self.environment = self.planner.environment
                self.yumi = self.environment.get_robot()
                self.yumi.set_speed(speed)
                # self.studio = Studio()
                self.studio = None
                self.visualize = False  # Visualizing robot motion in Jacobi Studio causes a pause in execution, so by default it's off
            else:
                raise FileNotFoundError(f"File {file} does not exist")
        else:
            self.yumi = Yumi()
            self.yumi.set_speed(speed)
            self.planner = Planner(self.yumi)
            self.environment = None
            self.studio = None
            self.visualize = False

        self.planner = Planner(self.yumi)

        module = ABBDriver.RapidModule(
            unit="ROB_L", egm_config="default", max_speed_deviation=100.0
        )
        module.upload = False
        self.driver_left = YuMiArm(
            self.planner,
            self.yumi.left,
            host="192.168.125.1",
            port=6511,
            module=module,
            version=ABBDriver.RobotWareVersion.RobotWare6,
            on_fail=on_fail
        )

        module = ABBDriver.RapidModule(
            unit="ROB_R", egm_config="default", max_speed_deviation=100.0
        )
        module.upload = False
        self.driver_right = YuMiArm(
            self.planner,
            self.yumi.right,
            host="192.168.125.1",
            port=6512,
            module=module,
            version=ABBDriver.RobotWareVersion.RobotWare6,
            on_fail=on_fail
        )
        self.driver_left.path_deviation_error_threshold = 0.3

        self.driver_right.path_deviation_error_threshold = 0.3

        self.set_TCP()
        self.print_timing = False

    def set_TCP(self):
        self.yumi.left.flange_to_tcp = self.RT2Frame(self.l_tcp)
        self.yumi.right.flange_to_tcp = self.RT2Frame(self.r_tcp)

    def get_FK(self, arm: Literal["left", "right"]):
        if arm == "left":
            return self.Frame2RT(
                self.yumi.left.calculate_tcp(self.driver_left.current_joint_position)
            )
        elif arm == "right":
            return self.Frame2RT(
                self.yumi.right.calculate_tcp(self.driver_right.current_joint_position)
            )

    def get_joint_positions(self, arm: Literal["left", "right"]):
        if arm == "left":
            return self.driver_left.current_joint_position
        elif arm == "right":
            return self.driver_right.current_joint_position

    async def stationary_signal_listener(self):
        while True:
            if ~int(
                self.driver_left._get_signal("OUTPUT_STATIONARY_ROB_L").lvalue
            ) or ~(self.driver_right._get_signal("OUTPUT_STATIONARY_ROB_R").lvalue):
                t = time.time()
                if self.print_timing:
                    print("Robot is no longer stationary at time: ", t)
                return t
            await asyncio.sleep(0.01)

    async def calibrate_grippers(self):
        await self.driver_left.calibrate_gripper(sync=True)
        await asyncio.sleep(0.1)
        await self.driver_right.calibrate_gripper()
        await asyncio.sleep(1)
        await self.driver_left.open_gripper(sync=True)
        await asyncio.sleep(0.1)
        await self.driver_right.open_gripper()
        await asyncio.sleep(1)
        return True

    async def close_grippers(self, side: Literal["both", "left", "right"] = "both"):
        if side == "both":
            await self.driver_left.close_gripper(sync=True)
            await asyncio.sleep(0.1)
            await self.driver_right.close_gripper()
            await asyncio.sleep(1)
        elif side == "left":
            await self.driver_left.close_gripper()
            await asyncio.sleep(0.5)
        elif side == "right":
            await self.driver_right.close_gripper()
            await asyncio.sleep(0.5)
        return True

    async def open_grippers(self, side: Literal["both", "left", "right"] = "both"):
        if side == "both":
            await self.driver_left.open_gripper(sync=True)
            await asyncio.sleep(0.1)
            await self.driver_right.open_gripper()
            await asyncio.sleep(1)
        elif side == "left":
            await self.driver_left.open_gripper()
            await asyncio.sleep(0.5)
        elif side == "right":
            await self.driver_right.open_gripper()
            await asyncio.sleep(0.5)
        return True

    async def grippers_move_to(self, left_dist=0, right_dist=0):
        await self.driver_left.move_gripper(left_dist, sync=True)
        await asyncio.sleep(0.1)
        await self.driver_right.move_gripper(right_dist)
        await asyncio.sleep(1)
        return True

    async def home(self):
        l_pos = self.yumi.left.calculate_tcp(self.driver_left.current_joint_position)
        r_pos = self.yumi.right.calculate_tcp(self.driver_right.current_joint_position)
        self.yumi.set_speed(0.2)
        if l_pos.translation[2] < 0.08 or r_pos.translation[2] < 0.08:
            await self.go_delta(
                [0, 0, 0.30 - l_pos.translation[2]], [0, 0, 0.30 - r_pos.translation[2]]
            )

        result_left, result_right = await self.move_to(
            list(self.L_ARMS_CLEAR_STATE), list(self.R_ARMS_CLEAR_STATE)
        )
        self.yumi.set_speed(self.speed)
        print("Homing done")
        return result_left, result_right

    async def home2(self):
        l_pos = self.yumi.left.calculate_tcp(self.driver_left.current_joint_position)
        r_pos = self.yumi.right.calculate_tcp(self.driver_right.current_joint_position)
        self.yumi.set_speed(0.2)
        if l_pos.translation[2] < 0.08 or r_pos.translation[2] < 0.08:
            await self.go_delta(
                [0, 0, 0.30 - l_pos.translation[2]], [0, 0, 0.30 - r_pos.translation[2]]
            )

        result_left, result_right = await self.move_to(
            list(self.L_HOME_STATE), list(self.R_HOME_STATE)
        )
        self.yumi.set_speed(self.speed)
        print("Homing done")
        return result_left, result_right

    async def move_to(self, left_goal=None, right_goal=None, ignore_collisions=False):
        result_left, result_right = None, None
        if left_goal is not None:
            result_left = self.driver_left.move_to_async(
                left_goal, ignore_collisions=ignore_collisions
            )
        if right_goal is not None:
            result_right = self.driver_right.move_to_async(
                right_goal, ignore_collisions=ignore_collisions
            )
        if left_goal is not None:
            await result_left
        if right_goal is not None:
            await result_right
        return result_left, result_right

    async def go_delta(
        self, left_delta: List = [0, 0, 0], right_delta: List = [0, 0, 0]
    ):
        assert (
            len(left_delta) == 3 and len(right_delta) == 3
        ), "Delta must be a list of 3 elements"
        l_curr_pos = self.yumi.left.calculate_tcp(
            self.driver_left.current_joint_position
        )
        l_goal_mat = l_curr_pos.matrix
        l_goal_mat[-4:-1] += np.array(left_delta)
        l_goal = Frame.from_matrix(l_goal_mat)

        r_curr_pos = self.yumi.right.calculate_tcp(
            self.driver_right.current_joint_position
        )
        r_goal_mat = r_curr_pos.matrix
        r_goal_mat[-4:-1] += np.array(right_delta)
        r_goal = Frame.from_matrix(r_goal_mat)

        start_time = time.time()
        trajectory = self.planner.plan(
            start={
                self.yumi.left: self.driver_left.current_joint_position,
                self.yumi.right: self.driver_right.current_joint_position,
            },
            goal={
                self.yumi.left: CartesianWaypoint(l_goal),
                self.yumi.right: CartesianWaypoint(r_goal),
            },
        )
        if self.print_timing:
            elapsed_time = (time.time() - start_time) * 1000
            print(f"[go_delta] Time to plan dual arm deltas: {elapsed_time:.3f} ms")

        if self.studio is not None and self.visualize:
            self.studio.run_trajectory(trajectory)
        if self.print_timing:
            start_time = time.time()
            print(
                "[go_delta] Sent trajectory and starting signal listener at time: ",
                start_time,
            )

        result_left = self.driver_left.run_async(trajectory)
        result_right = self.driver_right.run_async(trajectory)

        result_stationary = await self.stationary_signal_listener()
        if self.print_timing:
            print(
                f"[go_delta] Time to deploy: {(result_stationary - start_time)*1000} ms"
            )

        await result_left
        await result_right
        return result_left, result_right

    async def go_cartesian_waypoints(
        self, l_targets: List[RigidTransform] = [], r_targets: List[RigidTransform] = []
    ):
        assert len(l_targets) > 0 or len(r_targets) > 0, "No waypoints provided"
        result_left, result_right = None, None
        if len(l_targets) > 0:
            motion = self.listRT2Motion(
                robot=self.yumi.left,
                start=self.driver_left.current_joint_position,
                wp_list=l_targets,
            )
            start_time = time.time()
            trajectory_l = self.planner.plan(motion)
            if self.print_timing:
                elapsed_time = (time.time() - start_time) * 1000
                print(
                    f"[go_cartesian_waypoints] Time to plan left arm waypoints: {elapsed_time:.3f} ms"
                )

        if len(r_targets) > 0:
            motion = self.listRT2Motion(
                robot=self.yumi.right,
                start=self.driver_right.current_joint_position,
                wp_list=r_targets,
            )
            start_time = time.time()
            trajectory_r = self.planner.plan(motion)
            if self.print_timing:
                elapsed_time = (time.time() - start_time) * 1000
                print(
                    f"[go_cartesian_waypoints] Time to plan right arm waypoints: {elapsed_time:.3f} ms"
                )

        if self.studio is not None and self.visualize:
            self.studio.run_trajectory(trajectory_l)
            self.studio.run_trajectory(trajectory_r)

        if self.print_timing:
            start_time = time.time()
            print(
                "[go_cartesian_waypoints] Sent trajectory and starting signal listener at time: ",
                start_time,
            )

        if len(l_targets) > 0:
            result_left = self.driver_left.run_async(trajectory_l)
        if len(r_targets) > 0:
            result_right = self.driver_right.run_async(trajectory_r)

        if self.print_timing:
            result_stationary = await self.stationary_signal_listener()
            print(
                f"[go_cartesian_waypoints] Time to deploy: {(result_stationary - start_time)*1000} ms"
            )

        if len(l_targets) > 0:
            await result_left
        if len(r_targets) > 0:
            await result_right
        return result_left, result_right

    async def go_linear_single(
        self, l_target: RigidTransform = None, r_target: RigidTransform = None
    ):
        assert (
            l_target is not None or r_target is not None
        ), "No linear targets provided"
        result_left, result_right = None, None
        if l_target is not None:
            motion = LinearMotion(
                robot=self.yumi.left,
                start=self.driver_left.current_joint_position,
                goal=self.RT2Frame(l_target),
            )
            start_time = time.time()
            trajectory_l = self.planner.plan(motion)
            if self.print_timing:
                elapsed_time = (time.time() - start_time) * 1000
                print(
                    f"[go_linear_single] Time to plan left arm linear: {elapsed_time:.3f} ms"
                )

        if r_target is not None:
            motion = LinearMotion(
                robot=self.yumi.right,
                start=self.driver_right.current_joint_position,
                goal=self.RT2Frame(r_target),
            )
            start_time = time.time()
            trajectory_r = self.planner.plan(motion)
            if self.print_timing:
                elapsed_time = (time.time() - start_time) * 1000
                print(
                    f"[go_linear_single] Time to plan right arm linear: {elapsed_time:.3f} ms"
                )

        if self.studio is not None and self.visualize:
            if l_target is not None:
                self.studio.run_trajectory(trajectory_l)
            if r_target is not None:
                self.studio.run_trajectory(trajectory_r)

        if self.print_timing:
            start_time = time.time()
            print(
                "[go_linear_single] Sent trajectory and starting signal listener at time: ",
                start_time,
            )

        if l_target is not None:
            result_left = self.driver_left.run_async(trajectory_l)
        if r_target is not None:
            result_right = self.driver_right.run_async(trajectory_r)

        if self.print_timing:
            result_stationary = await self.stationary_signal_listener()
            print(
                f"[go_linear_single] Time to deploy: {(result_stationary - start_time)*1000} ms"
            )

        if l_target is not None:
            await result_left
        if r_target is not None:
            await result_right
        return result_left, result_right

    def plan_cartesian_waypoints(
        self,
        l_targets: List[RigidTransform],
        r_targets: List[RigidTransform],
        start_from_current_cfg=True,
    ) -> List[Trajectory]:
        assert len(l_targets) > 0 or len(r_targets) > 0, "No waypoints provided"
        if len(l_targets) > 0:
            l_motion = self.listRT2Motion(
                robot=self.yumi.left,
                start=(
                    self.driver_left.current_joint_position
                    if start_from_current_cfg
                    else self.RT2CW(l_targets[0])
                ),
                wp_list=l_targets if start_from_current_cfg else l_targets[1:],
            )
        if len(r_targets) > 0:
            r_motion = self.listRT2Motion(
                robot=self.yumi.right,
                start=(
                    self.driver_right.current_joint_position
                    if start_from_current_cfg
                    else self.RT2CW(r_targets[0])
                ),
                wp_list=r_targets if start_from_current_cfg else r_targets[1:],
            )
        motion = [l_motion, r_motion]
        trajectory = self.planner.plan(motion)

        if self.print_timing:
            print(
                f"[plan_linear] Calculation duration: {self.planner.last_calculation_duration:0.2f} [ms]"
            )

        return trajectory

    def plan_linear_waypoints(
        self,
        l_targets: List[RigidTransform],
        r_targets: List[RigidTransform],
        start_from_current_cfg=True,
    ) -> List[Trajectory]:
        assert len(l_targets) > 0 or len(r_targets) > 0, "No waypoints provided"

        motions = []
        if len(l_targets) > 0:
            if start_from_current_cfg:
                motions.append(
                    LinearMotion(
                        robot=self.yumi.left,
                        start=self.driver_left.current_joint_position,
                        goal=self.RT2CW(l_targets[0]),
                    )
                )
            for i in range(len(l_targets) - 1):
                motions.append(
                    LinearMotion(
                        robot=self.yumi.left,
                        start=self.RT2CW(l_targets[i]),
                        goal=self.RT2CW(l_targets[i + 1]),
                    )
                )

        if len(r_targets) > 0:
            if start_from_current_cfg:
                motions.append(
                    LinearMotion(
                        robot=self.yumi.right,
                        start=self.driver_right.current_joint_position,
                        goal=self.RT2CW(r_targets[0]),
                    )
                )
            for i in range(len(r_targets) - 1):
                motions.append(
                    LinearMotion(
                        robot=self.yumi.right,
                        start=self.RT2CW(r_targets[i]),
                        goal=self.RT2CW(r_targets[i + 1]),
                    )
                )
        trajectory = self.planner.plan(motions)
        if self.print_timing:
            print(f"planning {len(motions)} motions")
            print(
                f"[plan_linear_waypoints] Calculation duration: {self.planner.last_calculation_duration:0.2f} [ms]"
            )

        return trajectory

    # def plan(self, motion: List[Motion]) -> List[Trajectory]:
    #     return self.planner.plan(motion)

    # def plan(self, motion: Motion) -> Trajectory:
    #     return self.planner.plan(motion)

    async def run_trajectory(
        self, l_trajectory: Trajectory = None, r_trajectory: Trajectory = None
    ):
        result_left, result_right = None, None
        if self.studio is not None and self.visualize:
            if l_trajectory is not None:
                self.studio.run_trajectory(l_trajectory)
            if r_trajectory is not None:
                self.studio.run_trajectory(r_trajectory)
        if l_trajectory is not None:
            result_left = self.driver_left.run_async(l_trajectory)
        if r_trajectory is not None:
            result_right = self.driver_right.run_async(r_trajectory)
        if l_trajectory is not None:
            await result_left
        if r_trajectory is not None:
            await result_right
        return result_left, result_right

    def run_trajectory_no_async(
        self, l_trajectory: Trajectory = None, r_trajectory: Trajectory = None
    ):
        result_left, result_right = None, None
        if self.studio is not None and self.visualize:
            if l_trajectory is not None:
                self.studio.run_trajectory(l_trajectory)
            if r_trajectory is not None:
                self.studio.run_trajectory(r_trajectory)
        if l_trajectory is not None:
            result_left = self.driver_left.run(l_trajectory)
        if r_trajectory is not None:
            result_right = self.driver_right.run(r_trajectory)
        return result_left, result_right

    async def run_trajectories(self, trajectories: List[Trajectory]):
        results_left, results_right = [], []
        for trajectory in trajectories:
            result_left, result_right = await self.run_trajectory(
                trajectory, trajectory
            )
            results_left.append(result_left)
            results_right.append(result_right)

        return results_left, results_right

    async def blend_into(
        self, trajectory_l: Motion, trajectory_r: Motion, duration=1.0
    ):
        if self.studio is not None and self.visualize:
            self.studio.run_trajectory(trajectory_l)
            self.studio.run_trajectory(trajectory_r)
        result_left = self.driver_left.blend_into_async(trajectory_l, duration=duration)
        result_right = self.driver_right.blend_into_async(
            trajectory_r, duration=duration
        )
        await result_left
        await result_right
        return result_left, result_right

    def listRT2Motion(
        self, robot, start: Union[List, RigidTransform], wp_list: List[RigidTransform]
    ) -> Motion:
        # Convert list of RigidTransform to Motion (Jacobi object)
        if isinstance(start, RigidTransform):
            start = self.RT2Frame(start)
        motion = Motion(robot=robot, start=start, goal=self.RT2Frame(wp_list[-1]))
        motion.waypoints = [self.RT2Frame(q) for q in wp_list[:-1]]
        return motion

    def listRT2LinearMotion(
        self, robot, start: Union[List, RigidTransform], goal: RigidTransform
    ) -> Motion:
        # Convert list of RigidTransform to Motion (Jacobi object)
        if isinstance(start, RigidTransform):
            start = self.RT2Frame(start)
        motion = LinearMotion(robot=robot, start=start, goal=self.RT2Frame(goal))
        return motion

    def RT2CW(self, transform: RigidTransform) -> CartesianWaypoint:
        # Convert RigidTransform (autolab_core) to CartesianWaypoint (Jacobi object)
        return CartesianWaypoint(self.RT2Frame(transform))

    def RT2Frame(self, transform: RigidTransform) -> Frame:
        # Convert RigidTransform (autolab_core) to Frame (Jacobi object)
        return Frame(
            x=transform.translation[0],
            y=transform.translation[1],
            z=transform.translation[2],
            qw=transform.quaternion[0],
            qx=transform.quaternion[1],
            qy=transform.quaternion[2],
            qz=transform.quaternion[3],
        )

    def Frame2RT(self, frame: Frame) -> RigidTransform:
        # Convert Frame (Jacobi object) to RigidTransform (autolab_core)
        return RigidTransform(
            translation=frame.translation,
            rotation=RigidTransform.rotation_from_quaternion(frame.quaternion),
        )
