import sys
import numpy as np
from yumi_jacobi.interface import Interface
from autolab_core import RigidTransform
from cable_routing.configs.envconfig import ExperimentConfig
from typing import Literal, Optional, List, Tuple
import copy

class YuMiRobotEnv:
    """ Wrapper on top of Justin's class for the YuMi robot environment. """

    def __init__(self):
        print("[YUMI_JACOBI] Initializing YuMi...")
        robot_config = ExperimentConfig.robot_cfg
        self.interface = Interface(speed=0.5)
        self.interface.yumi.left.min_position = robot_config.YUMI_MIN_POS
        self.interface.yumi.right.min_position = robot_config.YUMI_MIN_POS
        self.move_to_home()
        self.close_grippers()
        print("[YUMI_JACOBI] Done initializing YuMi.")

    def move_to_home(self) -> None:
        """ Moves both arms to the home position. """
        self.interface.home()

    def close_grippers(self) -> None:
        """ Closes both grippers. """
        self.interface.close_grippers()

    def plan_linear_waypoints(
        self,
        arms: Literal["left", "right", "both"],
        start_pose_l: Optional[RigidTransform] = None,
        end_pose_l: Optional[RigidTransform] = None,
        start_pose_r: Optional[RigidTransform] = None,
        end_pose_r: Optional[RigidTransform] = None
    ) -> List:
        """ Plans linear waypoints for one or both arms. """
        return self.interface.plan_linear_waypoints(
            l_targets=[start_pose_l, end_pose_l] if arms in ["left", "both"] else [],
            r_targets=[start_pose_r, end_pose_r] if arms in ["right", "both"] else []
        )

    def execute_trajectory(
    self, l_trajectory: Optional = None, r_trajectory: Optional = None
    ) -> None:
        """ Executes planned trajectory for one or both arms. """
        self.interface.run_trajectory(l_trajectory, r_trajectory)
        
    def plan_and_execute_linear_waypoints(
        self,
        arms: Literal["left", "right", "both"],
        start_pose_l: Optional[RigidTransform] = None,
        end_pose_l: Optional[RigidTransform] = None,
        start_pose_r: Optional[RigidTransform] = None,
        end_pose_r: Optional[RigidTransform] = None
    ) -> None:
            
        # Plan the trajectory
        trajectories = self.interface.plan_linear_waypoints(
            l_targets=[start_pose_l, end_pose_l] if arms in ["left", "both"] else [],
            r_targets=[start_pose_r, end_pose_r] if arms in ["right", "both"] else []
        )

        # Execute the planned trajectory
        self.interface.run_trajectory(
            l_trajectory=trajectories[0] if arms in ["left", "both"] else None,
            r_trajectory=trajectories[1] if arms in ["right", "both"] else None
        )


    def set_joint_positions(
        self, 
        left_positions: Optional[List[float]] = None, 
        right_positions: Optional[List[float]] = None
    ) -> None:
        """ Moves arms to specific joint positions. """
        self.interface.move_to(left_goal=left_positions, right_goal=right_positions)

    def set_ee_pose(
        self, 
        left_pose: Optional[RigidTransform] = None, 
        right_pose: Optional[RigidTransform] = None
    ) -> None:
        """ Moves arms linearly to a specific end-effector pose. """
        self.interface.go_linear_single(l_target=left_pose, r_target=right_pose)

    def go_delta(
        self, 
        left_delta: Optional[Tuple[float, float, float]] = None, 
        right_delta: Optional[Tuple[float, float, float]] = None
    ) -> None:
        """ Moves arms by a specified cartesian delta. """
        self.interface.go_delta(left_delta=left_delta or [0, 0, 0], 
                                right_delta=right_delta or [0, 0, 0])

    def rotate_gripper(self, angle: float, arm: Literal["left", "right", "both"]) -> None:
        """ Rotates the wrist of the specified arm(s) in a single motion. """
        
        # Retrieve joint positions for both arms
        left_positions = self.interface.get_joint_positions("left") if arm in ["left", "both"] else None
        right_positions = self.interface.get_joint_positions("right") if arm in ["right", "both"] else None

        # Apply rotation if the arm is specified
        if left_positions:
            left_positions[-1] = np.clip(left_positions[-1] + angle, -2 * np.pi, 2 * np.pi)
        if right_positions:
            right_positions[-1] = np.clip(right_positions[-1] + angle, -2 * np.pi, 2 * np.pi)

        # Move both arms in a single command
        self.interface.move_to(left_goal=left_positions, right_goal=right_positions)


    def rotate_pose_by_rpy(
        self, pose: RigidTransform, roll: float, pitch: float, yaw: float
    ) -> RigidTransform:
        """ Rotates a given pose by roll, pitch, yaw in the parent coordinate system. """
        delta_rotation = RigidTransform.rotation_from_axis_angle([roll, pitch, yaw])
        new_rotation = pose.rotation @ delta_rotation
        return RigidTransform(rotation=new_rotation, translation=pose.translation)

    def go_delta_action(
        self, action_xyz: Optional[Tuple[float, float, float]] = None, 
        action_theta: Optional[Tuple[float, float, float]] = None
    ) -> None:
        """ Applies a position and orientation delta action to the end effector. """
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
        """ Sets the end effector pose using translation and quaternion rotation. """
        pose = RigidTransform(rotation=RigidTransform.rotation_from_quaternion(rot), translation=trans)
        self.set_ee_pose(left_pose=pose, right_pose=pose)

    def set_trajectory_joints(
        self, positions_array: List[List[float]]
    ) -> None:
        """ Sets the trajectory for the joints of both arms. """
        self.set_joint_positions(left_positions=positions_array[0], right_positions=positions_array[1])

    def get_ee_pose(self) -> Tuple[RigidTransform, RigidTransform]:
        """ Returns the end effector pose of both arms. """
        return self.interface.get_FK("left"), self.interface.get_FK("right")

    def get_ee_rpy(self) -> Tuple[List[float], List[float]]:
        """ Returns the roll-pitch-yaw of both end effectors. """
        left_pose, right_pose = self.get_ee_pose()
        return list(left_pose.euler_angles), list(right_pose.euler_angles)

    def get_joint_values(self) -> Tuple[List[float], List[float]]:
        """ Returns the current joint positions of both arms. """
        return self.interface.get_joint_positions("left"), self.interface.get_joint_positions("right")

    def get_jacobian_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """ Returns the Jacobian matrix of both arms. """
        # TODO
        pass


def main():
    """ Main function to test YuMiRobotEnv motions. """
    
    # Initialize the YuMi robot environment
    yumi_env = YuMiRobotEnv()

    # Print forward kinematics for both arms
    print("Left Arm FK:", yumi_env.get_ee_pose()[0])
    print("Right Arm FK:", yumi_env.get_ee_pose()[1])

    # Move to home position
    yumi_env.move_to_home()

    # Calibrate and open grippers
    yumi_env.interface.calibrate_grippers()
    yumi_env.interface.open_grippers()
    print("Left Gripper Position:", yumi_env.interface.driver_left.get_gripper_pos())
    print("Right Gripper Position:", yumi_env.interface.driver_right.get_gripper_pos())

    # Close grippers
    yumi_env.close_grippers()
    print("Left Gripper Position After Close:", yumi_env.interface.driver_left.get_gripper_pos())
    print("Right Gripper Position After Close:", yumi_env.interface.driver_right.get_gripper_pos())

    # Define waypoints for both arms
    wp1_l = RigidTransform(
        rotation=[[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
        translation=[0.4, 0.3, 0.2]
    )
    wp2_l = RigidTransform(
        rotation=[[-1.0000000, -0.0000000,  0.0000000], 
                  [-0.0000000,  0.9659258, -0.2588190], 
                  [-0.0000000, -0.2588190, -0.9659258]],
        translation=[0.4, 0.4, 0.1]
    )
    wp3_l = RigidTransform(
        rotation=[[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
        translation=[0.3, 0.3, 0.15]
    )
    wp1_r = RigidTransform(
        rotation=[[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
        translation=[0.3, -0.2, 0.15]
    )
    wp2_r = RigidTransform(
        rotation=[[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
        translation=[0.35, -0.05, 0.2]
    )
    wp3_r = RigidTransform(
        rotation=[[-0.7071068,  0.7071068,  0.0000000], 
                  [0.5000000,  0.5000000,  0.7071068], 
                  [0.5000000,  0.5000000, -0.7071068]],
        translation=[0.45, -0.08, 0.15]
    )

    # Plan and execute linear waypoints for both arms
    yumi_env.plan_and_execute_linear_waypoints(
        arms="both",
        start_pose_l=wp1_l, end_pose_l=wp2_l,
        start_pose_r=wp1_r, end_pose_r=wp2_r
    )

    yumi_env.plan_and_execute_linear_waypoints(
        arms="both",
        start_pose_l=wp2_l, end_pose_l=wp3_l,
        start_pose_r=wp2_r, end_pose_r=wp3_r
    )

    # Return home after motion execution
    yumi_env.move_to_home()

    # Test small incremental delta motions
    yumi_env.go_delta(left_delta=(0, 0, -0.1), right_delta=(0, 0, -0.1))
    yumi_env.go_delta(left_delta=(0, -0.1, 0), right_delta=(0, 0.1, 0))
    yumi_env.go_delta(left_delta=(0.1, -0.1, 0), right_delta=(0.1, 0.1, 0))

    # Move back to home
    yumi_env.move_to_home()


if __name__ == "__main__":
    main()
