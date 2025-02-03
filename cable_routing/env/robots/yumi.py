import sys
import numpy as np
from yumi_jacobi.interface import Interface
from autolab_core import RigidTransform, Point
from omegaconf import DictConfig
from cable_routing.configs.config import get_robot_config
from typing import Literal, Optional
import copy

class YuMiRobotEnv():
    """ Base class for the YuMi robot environment.
    """

    def __init__(self):
        
        print("Initializing YuMi...")
        robot_config = robot_properties = get_robot_config('yumi')
        self.interface = Interface(speed=0.5)
        self.interface.yumi.left.min_position  = robot_config.YUMI_MIN_POS
        self.interface.yumi.right.min_position = robot_config.YUMI_MIN_POS
        self.move_to_home()
        self.close_grippers()
        print("DONE")

    def move_to_home(self):
        
        self.interface.home()
    
    def close_grippers(self):
            
        self.interface.close_grippers()
            
    def plan_linear_waypoints(self,
                              arms:Literal["left", "right"],
                              start_pose:Optional[RigidTransform]=None,
                              end_pose:Optional[RigidTransform]=None):
        
        # TODO: fix for both arms..
        targets = [start_pose, end_pose]
        if "left" in arms:
            traj = self.interface.plan_linear_waypoints(l_targets=targets)
        elif "right" in arms:
            traj = self.interface.plan_linear_waypoints(r_targets=targets)
        else:
            raise ValueError("Invalid arm name. Please specify 'left' or 'right'.")
        
        return traj 

    def excute_trajectory(self, trajectory):
            
            self.interface.execute_trajectory(trajectory)
                 
    def set_joint_positions(self, arm:Literal["left", "right"], positions): 
        
        if "left" in arm:
            self.interface.move_to(right_goal= positions)
        elif "right" in arm:
            self.interface.move_to(left_goal = positions)
        else:
            raise ValueError("Invalid arm name. Please specify 'left' or 'right'.")
    
    def set_ee_pose(self, pose: RigidTransform , arm:Literal["left", "right"]):
        
        if "left" in arm:
            self.interface.go_linear_single(l_target=pose)
        elif "right" in arm:
            self.interface.go_linear_single(r_target=pose)
        else:
            raise ValueError("Invalid arm name. Please specify 'left' or 'right'.")
    
    def go_delta(self, delta_xyz, arm:Literal["left", "right"]):
            
        if "left" in arm:
            self.interface.go_delta(l_delta=delta_xyz)
        elif "right" in arm:
            self.interface.go_delta(r_delta=delta_xyz)
        else:
            raise ValueError("Invalid arm name. Please specify 'left' or 'right'.")
      
    def rotate_gripper(self, angle, arm:Literal["left", "right"]):
        
        current = self.interface.get_joint_positions(arm)
        negative = copy.deepcopy(current)

        current[-1] = min(current[-1] + angle, 2*np.pi)   # rotate wrist angles
        negative[-1] = max(current[-1] - angle, -2*np.pi)

        if arm == "right":
            self.interface.move_to(right_goal=current)
            self.interface.move_to(right_goal=negative)
        else:
            self.interface.move_to(left_goal=current)
            self.interface.move_to(left_goal=negative)
               
               
    ####################### TODO #############################
    def rotate_pose_by_rpy(self, in_pose, roll, pitch, yaw, wait=True):
        """
        Apply an RPY rotation to a pose in its parent coordinate system.
        """
        try:
            if in_pose.header:  # = in_pose is a PoseStamped instead of a Pose.
                in_pose.pose = self.rotate_pose_by_rpy(in_pose.pose, roll, pitch, yaw, wait)
                return in_pose
        except:
            pass
        q_in = [in_pose.orientation.x, in_pose.orientation.y, in_pose.orientation.z, in_pose.orientation.w]
        q_rot = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
        q_rotated = tf.transformations.quaternion_multiply(q_in, q_rot)

        rotated_pose = copy.deepcopy(in_pose)
        rotated_pose.orientation = geometry_msgs.msg.Quaternion(*q_rotated)
        result = self.move_manipulator.ee_traj_by_pose_target(rotated_pose, wait)
        return rotated_pose

    def apply_delta_action(self, action_xyz=None, action_theta=None, wait=True):
        """
        Sets the Pose of the EndEffector based on the action variable.
        The action variable contains the position and orientation of the EndEffector.
        See create_action
        """
        # Set up a trajectory message to publish.
        if action_theta is None:
            action_theta = [0.0, 0.0, 0.0]
        if action_xyz is None:
            action_theta = [0.0, 0.0, 0.0]

        action_xyz = action_xyz if isinstance(action_xyz, list) else action_xyz.tolist()
        if len(action_xyz) < 3: action_xyz.append(0.0)  # xy -> xyz
        ee_target = self.get_ee_pose(as_message=True)
        cur_ee_pose = self.get_ee_pose(as_message=True)

        if np.any(action_theta):
            roll, pitch, yaw = tf.transformations.euler_from_quaternion((cur_ee_pose.orientation.x,
                                                                         cur_ee_pose.orientation.y,
                                                                         cur_ee_pose.orientation.z,
                                                                         cur_ee_pose.orientation.w))
            roll = roll + action_theta[0] if action_theta[0] else roll
            pitch = pitch + action_theta[1] if action_theta[1] else pitch
            yaw = yaw + action_theta[2] if action_theta[2] else yaw
            quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
            ee_target.orientation.x, ee_target.orientation.y, ee_target.orientation.z, ee_target.orientation.w = quaternion

        if np.any(action_xyz):
            ee_target.position.x = cur_ee_pose.pose.position.x + action_xyz[0] if action_xyz[
                0] else cur_ee_pose.pose.position.x
            ee_target.position.y = cur_ee_pose.pose.position.y + action_xyz[1] if action_xyz[
                1] else cur_ee_pose.pose.position.y
            ee_target.position.z = cur_ee_pose.pose.position.z + action_xyz[2] if action_xyz[
                2] else cur_ee_pose.pose.position.z

        result = self.move_manipulator.ee_traj_by_pose_target(ee_target, wait)
        return result

    def set_ee_pose_from_trans_rot(self, trans, rot, wait=True):
        # Set up a trajectory message to publish.

        req_pose = self.get_ee_pose(as_message=True)

        req_pose.position.x = trans[0]
        req_pose.position.y = trans[1]
        req_pose.position.z = trans[2]
        req_pose.orientation.x = rot[0]
        req_pose.orientation.y = rot[1]
        req_pose.orientation.z = rot[2]
        req_pose.orientation.w = rot[3]

        result = self.set_ee_pose(req_pose, wait=wait)
        return result

    def set_trajectory_joints(self, positions_array, wait=True, by_moveit=True, by_vel=False):

        result = self.move_manipulator.joint_traj(positions_array, wait=wait, by_moveit=by_moveit, by_vel=by_vel)

        return result

    def get_ee_pose(self, rot_as_euler=False, as_message=False):
        """
        """
        if not as_message:
            gripper_pose = self.move_manipulator.ee_pose()
            x, y, z = gripper_pose.position.x, gripper_pose.position.y, gripper_pose.position.z

            if rot_as_euler:
                roll, pitch, yaw = tf.transformations.euler_from_quaternion((gripper_pose.orientation.x,
                                                                             gripper_pose.orientation.y,
                                                                             gripper_pose.orientation.z,
                                                                             gripper_pose.orientation.w))

                rot = [roll, pitch, yaw]
            else:
                rot = [gripper_pose.orientation.x,
                       gripper_pose.orientation.y,
                       gripper_pose.orientation.z,
                       gripper_pose.orientation.w]
                if rot[0] > 0:
                    rot = [-q for q in rot]
            return [x, y, z], rot
        else:
            self.move_manipulator.ee_pose()

    def get_ee_rpy(self):
        gripper_rpy = self.move_manipulator.ee_rpy()
        return gripper_rpy

    def get_joint_values(self):

        return self.move_manipulator.joint_values()

    def get_jacobian_matrix(self):

        return self.move_manipulator.get_jacobian_matrix()