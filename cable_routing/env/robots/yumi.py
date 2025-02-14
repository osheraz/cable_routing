import sys
import numpy as np
from yumi_jacobi.interface import Interface
from autolab_core import RigidTransform
from cable_routing.configs.envconfig import ExperimentConfig
from typing import Literal, Optional, List, Tuple, Union
import tyro
import time


class YuMiRobotEnv:
    def __init__(self, robot_config):
        print("[YUMI_JACOBI] Initializing YuMi...")

        self.robot_config = robot_config
        self.interface = Interface(speed=0.2)
        self.interface.yumi.left.min_position = robot_config.YUMI_MIN_POS
        self.interface.yumi.right.min_position = robot_config.YUMI_MIN_POS
        self.move_to_home()
        self.open_grippers()
        self.interface.calibrate_grippers()
        # self.close_grippers()
        print("[YUMI_JACOBI] Done initializing YuMi.")

    def move_to_home(self) -> None:
        """Moves both arms to the home position."""
        self.interface.home()

    def close_grippers(
        self, side: Literal["both", "left", "right"] = "both", wait: bool = False
    ):
        """Closes both grippers."""
        self.interface.close_grippers(side)
        if wait:
            time.sleep(1.0)

    def open_grippers(self, side: Literal["both", "left", "right"] = "both"):
        """Closes both grippers."""
        self.interface.open_grippers(side)

    def grippers_move_to(self, hand: Literal["left", "right", "both"], distance: int):
        """
        Move the specified gripper(s) to the given width [0, 25] (mm).

        Parameters:
        - hand (str): "left", "right", or "both" to specify which gripper(s) to move.
        - distance (int): Desired gripper opening width in mm.
        """
        left, right = self.get_gripper_pose("both")
        distance = int(distance)

        if hand == "left":
            self.interface.grippers_move_to(left_dist=distance, right_dist=right)
        elif hand == "right":
            self.interface.grippers_move_to(left_dist=left, right_dist=distance)
        elif hand == "both":
            self.interface.grippers_move_to(left_dist=distance, right_dist=distance)
        else:
            raise ValueError(
                "Invalid hand argument. Choose from 'left', 'right', or 'both'."
            )

    def get_gripper_pose(
        self, hand: Literal["left", "right", "both"]
    ) -> Union[int, Tuple[int, int]]:
        """
        Get the current gripper position.

        Parameters:
        - hand (str): "left", "right", or "both" to specify which gripper(s) to retrieve.

        Returns:
        - int: If querying a single gripper, returns its position in mm.
        - Tuple[int, int]: If querying both, returns a tuple (left_gripper, right_gripper).
        """
        if hand == "left":
            return int(self.interface.driver_left.get_gripper_pos()) * 1000
        elif hand == "right":
            return int(self.interface.driver_right.get_gripper_pos()) * 1000
        elif hand == "both":
            return (
                int(self.interface.driver_left.get_gripper_pos()) * 1000,
                int(self.interface.driver_right.get_gripper_pos()) * 1000,
            )
        else:
            raise ValueError(
                "Invalid hand argument. Choose from 'left', 'right', or 'both'."
            )

    def plan_linear_waypoints(
        self,
        arms: Literal["left", "right", "both"],
        start_pose_l: Optional[RigidTransform] = None,
        end_pose_l: Optional[RigidTransform] = None,
        start_pose_r: Optional[RigidTransform] = None,
        end_pose_r: Optional[RigidTransform] = None,
    ) -> List:
        """Plans linear waypoints for one or both arms."""
        return self.interface.plan_linear_waypoints(
            l_targets=[start_pose_l, end_pose_l] if arms in ["left", "both"] else [],
            r_targets=[start_pose_r, end_pose_r] if arms in ["right", "both"] else [],
        )

    def execute_trajectory(
        self, l_trajectory: Optional = None, r_trajectory: Optional = None
    ) -> None:
        """Executes planned trajectory for one or both arms."""
        self.interface.run_trajectory(l_trajectory, r_trajectory)

    def plan_and_execute_linear_waypoints(
        self,
        arms: Literal["left", "right", "both"],
        start_pose_l: Optional[RigidTransform] = None,
        end_pose_l: Optional[RigidTransform] = None,
        start_pose_r: Optional[RigidTransform] = None,
        end_pose_r: Optional[RigidTransform] = None,
    ) -> None:

        # Plan the trajectory
        trajectories = self.interface.plan_linear_waypoints(
            l_targets=[start_pose_l, end_pose_l] if arms in ["left", "both"] else [],
            r_targets=[start_pose_r, end_pose_r] if arms in ["right", "both"] else [],
        )

        # Execute the planned trajectory
        self.interface.run_trajectory(
            l_trajectory=trajectories[0] if arms in ["left", "both"] else None,
            r_trajectory=trajectories[1] if arms in ["right", "both"] else None,
        )

    def set_joint_positions(
        self,
        left_positions: Optional[List[float]] = None,
        right_positions: Optional[List[float]] = None,
    ) -> None:
        """Moves arms to specific joint positions."""
        self.interface.move_to(left_goal=left_positions, right_goal=right_positions)

    def set_ee_pose(
        self,
        left_pose: Optional[RigidTransform] = None,
        right_pose: Optional[RigidTransform] = None,
    ) -> None:
        """Moves arms linearly to a specific end-effector pose."""
        self.interface.go_linear_single(l_target=left_pose, r_target=right_pose)

    def go_delta(
        self,
        left_delta: Optional[Tuple[float, float, float]] = None,
        right_delta: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """Moves arms by a specified cartesian delta."""
        self.interface.go_delta(
            left_delta=left_delta or [0, 0, 0], right_delta=right_delta or [0, 0, 0]
        )

    def rotate_gripper(
        self, angle: float, arm: Literal["left", "right", "both"]
    ) -> None:
        """Rotates the wrist of the specified arm(s) in a single motion."""

        # Retrieve joint positions for both arms
        left_positions = (
            self.interface.get_joint_positions("left")
            if arm in ["left", "both"]
            else None
        )
        right_positions = (
            self.interface.get_joint_positions("right")
            if arm in ["right", "both"]
            else None
        )

        # Apply rotation if the arm is specified
        if left_positions:
            left_positions[-1] = np.clip(
                left_positions[-1] + angle, -2 * np.pi, 2 * np.pi
            )
        if right_positions:
            right_positions[-1] = np.clip(
                right_positions[-1] + angle, -2 * np.pi, 2 * np.pi
            )

        # Move both arms in a single command
        self.interface.move_to(left_goal=left_positions, right_goal=right_positions)

    def rotate_pose_by_rpy(
        self, pose: RigidTransform, roll: float, pitch: float, yaw: float
    ) -> RigidTransform:
        """Rotates a given pose by roll, pitch, yaw in the parent coordinate system."""
        delta_rotation = RigidTransform.rotation_from_axis_angle([roll, pitch, yaw])
        new_rotation = pose.rotation @ delta_rotation
        return RigidTransform(rotation=new_rotation, translation=pose.translation)

    def go_delta_action(
        self,
        action_xyz: Optional[Tuple[float, float, float]] = None,
        action_theta: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """Applies a position and orientation delta action to the end effector."""
        action_xyz = action_xyz or (0.0, 0.0, 0.0)
        action_theta = action_theta or (0.0, 0.0, 0.0)

        left_pose, right_pose = self.get_ee_pose()

        new_left_pose = self.rotate_pose_by_rpy(left_pose, *action_theta)
        new_left_pose.translation += np.array(action_xyz)

        new_right_pose = self.rotate_pose_by_rpy(right_pose, *action_theta)
        new_right_pose.translation += np.array(action_xyz)

        self.set_ee_pose(left_pose=new_left_pose, right_pose=new_right_pose)

    def set_ee_pose_from_trans_rot(
        self, trans: Tuple[float, float, float], rot: Tuple[float, float, float, float]
    ) -> None:
        """Sets the end effector pose using translation and quaternion rotation."""
        pose = RigidTransform(
            rotation=RigidTransform.rotation_from_quaternion(rot), translation=trans
        )
        self.set_ee_pose(left_pose=pose, right_pose=pose)

    def set_trajectory_joints(self, positions_array: List[List[float]]) -> None:
        """Sets the trajectory for the joints of both arms."""
        self.set_joint_positions(
            left_positions=positions_array[0], right_positions=positions_array[1]
        )

    def get_ee_pose(self) -> Tuple[RigidTransform, RigidTransform]:
        """Returns the end effector pose of both arms."""
        return self.interface.get_FK("left"), self.interface.get_FK("right")

    def get_ee_rpy(self) -> Tuple[List[float], List[float]]:
        """Returns the roll-pitch-yaw of both end effectors."""
        left_pose, right_pose = self.get_ee_pose()
        return list(left_pose.euler_angles), list(right_pose.euler_angles)

    def get_joint_values(self) -> Tuple[List[float], List[float]]:
        """Returns the current joint positions of both arms."""
        return self.interface.get_joint_positions(
            "left"
        ), self.interface.get_joint_positions("right")

    def get_jacobian_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the Jacobian matrix of both arms."""
        # TODO
        pass

    def single_hand_grasp(
        self, world_coord: np.ndarray, slow_mode: bool = False
    ) -> None:
        """ """

        arm = "right" if world_coord[1] < 0 else "left"

        print(f"Moving {arm} arm to the target position.")

        # Move above the target
        target_pose = RigidTransform(
            rotation=RigidTransform.x_axis_rotation(np.pi), translation=world_coord
        )

        target_pose.translation[2] += 0.1

        self.set_ee_pose(
            left_pose=target_pose if arm == "left" else None,
            right_pose=target_pose if arm == "right" else None,
        )

        # Move down in two steps for a smooth approach
        for _ in range(2):
            target_pose.translation[2] -= 0.05
            self.set_ee_pose(
                left_pose=target_pose if arm == "left" else None,
                right_pose=target_pose if arm == "right" else None,
            )
            # TODO wrap this function
            original_speed = self.interface._async_interface.speed
            if slow_mode:
                self.interface.yumi.set_speed(original_speed * 0.1)

        # Close gripper to grasp the object
        self.close_grippers(side=arm, wait=True)
        self.interface.yumi.set_speed(original_speed)

        # target_pose.translation[2] += 0.05
        # self.set_ee_pose(
        #     left_pose=target_pose if arm == "left" else None,
        #     right_pose=target_pose if arm == "right" else None,
        # )

        print(f"{arm.capitalize()} arm grasp completed.")

    def dual_hand_grasp(
        self,
        world_coord: Union[np.ndarray, List[np.ndarray]],
        spacing: float = 0.18,
        axis: Literal["x", "y"] = "y",
        slow_mode: bool = False,
    ) -> None:

        axis_index = 0 if axis == "x" else 1

        if isinstance(world_coord, list) and len(world_coord) == 2:
            world_coord_left, world_coord_right = world_coord
        else:
            world_coord_left = np.copy(world_coord)
            world_coord_right = np.copy(world_coord)
            world_coord_left[axis_index] += spacing / 2
            world_coord_right[axis_index] -= spacing / 2

        if abs(world_coord_left[axis_index] - world_coord_right[axis_index]) < 0.1:
            print("Error: Left and right arm poses would collide. Adjust positions.")
            print("Left:", world_coord_left)
            print("Right:", world_coord_right)
            return

        rot = (
            RigidTransform.y_axis_rotation(np.pi)
            if axis_index
            else RigidTransform.y_axis_rotation(np.pi)
            @ RigidTransform.z_axis_rotation(np.pi / 2)
        )
        target_pose_left = RigidTransform(rotation=rot, translation=world_coord_left)
        target_pose_right = RigidTransform(
            rotation=rot,
            translation=world_coord_right,
        )

        target_pose_left.translation[2] += 0.1
        target_pose_right.translation[2] += 0.1

        self.set_ee_pose(left_pose=target_pose_left, right_pose=target_pose_right)

        original_speed = self.interface._async_interface.speed
        if slow_mode:
            self.interface.yumi.set_speed(original_speed * 0.1)

        for _ in range(2):
            target_pose_left.translation[2] -= 0.05
            target_pose_right.translation[2] -= 0.05
            self.set_ee_pose(left_pose=target_pose_left, right_pose=target_pose_right)

        self.close_grippers(side="both", wait=True)

        target_pose_left.translation[2] += 0.1
        target_pose_right.translation[2] += 0.1
        self.set_ee_pose(left_pose=target_pose_left, right_pose=target_pose_right)
        self.interface.yumi.set_speed(original_speed)

        print("Dual-hand grasp completed.")

    def move_dual_hand_to(
        self,
        target_center: np.ndarray,
        slow_mode: bool = False,
    ) -> None:

        current_left, current_right = self.get_ee_pose()
        displacement_vector = (current_left.translation - current_right.translation) / 2
        new_left = target_center + displacement_vector
        new_right = target_center - displacement_vector

        original_speed = self.interface._async_interface.speed
        if slow_mode:
            self.interface.yumi.set_speed(original_speed * 0.1)

        target_pose_left = RigidTransform(
            rotation=current_left.rotation, translation=new_left
        )
        target_pose_right = RigidTransform(
            rotation=current_right.rotation, translation=new_right
        )

        self.set_ee_pose(left_pose=target_pose_left, right_pose=target_pose_right)
        self.interface.yumi.set_speed(original_speed)

    def move_dual_hand_insertion(
        self,
        target_center: np.ndarray,
        insertion_depth: float = 0.05,
        insertion_axis: Literal["x", "y", "z"] = "z",
        slow_mode: bool = False,
    ) -> None:
        self.move_dual_hand_to(target_center, slow_mode)

        print("Both hands have reached the target. Performing insertion...")

        axis_index = {"x": 0, "y": 1, "z": 2}[insertion_axis]
        target_center[axis_index] -= insertion_depth

        self.move_dual_hand_to(target_center, slow_mode=True)

        print(
            f"Insertion completed along {insertion_axis}-axis by {insertion_depth} meters."
        )

    def slide_hand(
        self,
        arm: Literal["left", "right"],
        axis: Literal["x", "y", "z"],
        amount: float = 0.1,
        gripper_opening: float = 5,
    ) -> None:
        """
        Slides the specified arm along the given axis while slightly opening the gripper.

        Parameters:
        - arm (str): "left" or "right" indicating which hand to slide.
        - axis (str): "x", "y", or "z" indicating the axis to slide along.
        - amount (float): The distance to slide in meters.
        - gripper_opening (float): How much to slightly open the gripper before sliding.
        """

        print(f"Sliding {arm} hand along {axis}-axis by {amount} meters.")

        current_pose = self.interface.get_FK(arm)
        cur_gripper_pos = self.get_gripper_pose(arm)  # close 0 - 25 open

        self.grippers_move_to(arm, distance=cur_gripper_pos + gripper_opening)
        time.sleep(0.5)

        axis_index = {"x": 0, "y": 1, "z": 2}[axis]
        current_pose.translation[axis_index] += amount

        if arm == "left":
            self.set_ee_pose(left_pose=current_pose)
        else:
            self.set_ee_pose(right_pose=current_pose)

        time.sleep(0.5)  # Wait before closing
        self.interface.close_grippers(side=arm)


def main(args: ExperimentConfig):
    """Main function to test all YuMiRobotEnv functionalities."""

    # Initialize the YuMi robot environment
    yumi_env = YuMiRobotEnv(args.robot_cfg)

    print("\n--- Initializing YuMi ---\n")
    yumi_env.move_to_home()
    yumi_env.open_grippers()

    yumi_env.grippers_move_to("left", 5)
    # Print FK for both arms
    print("\n--- Forward Kinematics ---")
    left_pose, right_pose = yumi_env.get_ee_pose()
    print("Left Arm FK:", left_pose)
    print("Right Arm FK:", right_pose)

    # Calibrate and open grippers
    print("\n--- Calibrating & Opening Grippers ---")
    yumi_env.interface.calibrate_grippers()
    yumi_env.interface.open_grippers()
    print("Left Gripper Position:", yumi_env.interface.driver_left.get_gripper_pos())
    print("Right Gripper Position:", yumi_env.interface.driver_right.get_gripper_pos())

    # Close grippers
    print("\n--- Closing Grippers ---")
    yumi_env.close_grippers()
    print(
        "Left Gripper Position After Close:",
        yumi_env.interface.driver_left.get_gripper_pos(),
    )
    print(
        "Right Gripper Position After Close:",
        yumi_env.interface.driver_right.get_gripper_pos(),
    )

    # Get joint values
    print("\n--- Current Joint Positions ---")
    left_joints, right_joints = yumi_env.get_joint_values()
    print("Left Joint Positions:", left_joints)
    print("Right Joint Positions:", right_joints)

    # Define waypoints for testing
    wp1_l = RigidTransform(
        rotation=[[-1, 0, 0], [0, 1, 0], [0, 0, -1]], translation=[0.4, 0.3, 0.2]
    )
    wp2_l = RigidTransform(
        rotation=[[-1.0, 0.0, 0.0], [0.0, 0.9659, -0.2588], [0.0, -0.2588, -0.9659]],
        translation=[0.4, 0.4, 0.1],
    )
    wp1_r = RigidTransform(
        rotation=[[-1, 0, 0], [0, 1, 0], [0, 0, -1]], translation=[0.3, -0.2, 0.15]
    )
    wp2_r = RigidTransform(
        rotation=[[-1, 0, 0], [0, 1, 0], [0, 0, -1]], translation=[0.35, -0.05, 0.2]
    )

    print("\n--- Planning and Executing Linear Waypoints ---")
    current_pose_l, current_pose_r = yumi_env.get_ee_pose()

    # Plan and execute waypoints starting from the current pose
    yumi_env.plan_and_execute_linear_waypoints(
        arms="both",
        start_pose_l=current_pose_l,
        end_pose_l=wp2_l,
        start_pose_r=current_pose_r,
        end_pose_r=wp2_r,
    )

    # Test moving delta
    print("\n--- Moving by Delta ---")
    yumi_env.go_delta(left_delta=(0.05, 0, 0), right_delta=(-0.05, 0, 0))

    # Test setting joint positions
    print("\n--- Setting Joint Positions ---")
    left_new_joints = np.clip(
        np.array(left_joints) + 0.1, -2 * np.pi, 2 * np.pi
    ).tolist()
    right_new_joints = np.clip(
        np.array(right_joints) - 0.1, -2 * np.pi, 2 * np.pi
    ).tolist()
    yumi_env.set_joint_positions(
        left_positions=left_new_joints, right_positions=right_new_joints
    )

    # Test rotating grippers
    print("\n--- Rotating Grippers ---")
    yumi_env.rotate_gripper(angle=np.pi / 6, arm="both")

    # Test setting end-effector pose
    print("\n--- Setting End-Effector Pose ---")
    new_pose_l = RigidTransform(rotation=wp2_l.rotation, translation=[0.35, 0.3, 0.15])
    new_pose_r = RigidTransform(rotation=wp2_r.rotation, translation=[0.3, -0.1, 0.2])
    yumi_env.set_ee_pose(left_pose=new_pose_l, right_pose=new_pose_r)

    # Test applying delta action
    print("\n--- Applying Delta Action ---")
    yumi_env.go_delta_action(action_xyz=(0, 0, 0.05), action_theta=(0, 0, np.pi / 12))

    # Test setting pose using translation & rotation
    print("\n--- Setting EE Pose Using Translation and Rotation ---")
    yumi_env.set_ee_pose_from_trans_rot(trans=(0.4, 0.3, 0.15), rot=(0, 0, 0, 1))

    # Test retrieving RPY
    print("\n--- Getting End Effector RPY ---")
    left_rpy, right_rpy = yumi_env.get_ee_rpy()
    print("Left EE RPY:", left_rpy)
    print("Right EE RPY:", right_rpy)

    # Test retrieving Jacobian
    print("\n--- Getting Jacobian Matrices ---")
    try:
        left_jacobian, right_jacobian = yumi_env.get_jacobian_matrix()
        print("Left Jacobian:\n", left_jacobian)
        print("Right Jacobian:\n", right_jacobian)
    except NotImplementedError:
        print("Jacobian computation not implemented yet.")

    print("\n--- All Tests Completed Successfully ---")


if __name__ == "__main__":

    args = tyro.cli(ExperimentConfig)

    main(args)
