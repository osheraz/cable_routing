# Copyright 2024 Nimrod Curtis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard library imports
import os
from dataclasses import dataclass, field
from typing import Any, List
import json

# Third party library imports
import numpy as np
import cv2
import matplotlib
from matplotlib import animation
import moviepy.video.io.VideoFileClip as mp

# ROS libraries
from cv_bridge import CvBridge, CvBridgeError
bridge = CvBridge()
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CompressedImage
from zed_interfaces.msg import *

class TrajViz():
    """
    A visualization utility class for creating trajectory animations.
    
    This class provides static methods to generate single frames for trajectory videos and to save those animations
    into MP4.
    
    Methods:
        visualization: Generates and returns graphical elements for a single frame of the trajectory animation.
        save_animation: Saves a Matplotlib animation object to a file in MP4 format.
    """
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def visualization(robot_position,
                    yaw,
                    curr_image,
                    time,
                    frame_idx,
                    ax_image,
                    ax_trajectory,
                    target_position=None,
                    corners=None,
                    goal_position=None,
                    goal_yaw=None) -> List[matplotlib.axes.Axes]:
        """
        Generates graphical elements for a single frame of the trajectory animation.

        Args:
            robot_position (np.array): The array of robot positions (x, y coordinates).
            yaw (float): The orientation of the robot.
            curr_image (np.array): The current RGB image from the video sequence.
            time (float): The timestamp of the current frame.
            frame_idx (int): The index of the current frame.
            ax_image (matplotlib.axes.Axes): The axes object for image plotting.
            ax_trajectory (matplotlib.axes.Axes): The axes object for trajectory plotting.
            target_position (np.array, optional): The array of target positions (x, y coordinates).
            corners (np.array, optional): The array of corner points of the target bounding box.

        Returns:
            list: A list of matplotlib artists (elements) that represent the current frame's graphical content.
        """
        Frame = []

        # Only add image elements if both curr_image and ax_image are provided
        if curr_image is not None and ax_image is not None:
            # Swap red and blue channels for correct color display in Matplotlib
            curr_image_red_blue_swapped = curr_image[:, :, ::-1]

            # Initialize frame list with image and text elements
            title = ax_image.text((curr_image.shape[1] - 100), 0, "", 
                                  bbox={'facecolor': 'w', 'alpha': 0.7, 'pad': 5},
                                  fontsize=12, ha="left", va="top")
            title.set_text(f"Frame: {frame_idx} | Time: {time:.4f} [sec]")
            Frame.append(title)
            Frame.append(ax_image.imshow(curr_image_red_blue_swapped))

        # Plot robot positions on the trajectory plot
        handles = []
        robot_pos_plot = ax_trajectory.plot(robot_position[:, 1], robot_position[:, 0],
                                            color="magenta", linewidth=4, markersize=8,
                                            label='Robot position [m]')[0]
        handles.append(robot_pos_plot)
        Frame.append(robot_pos_plot)

        # Plot target positions if available
        if target_position is not None and len(target_position) > 0:
            target_pos_plot = ax_trajectory.plot(target_position[:, 1], target_position[:, 0],
                                                 color="blue", linewidth=4, markersize=16,
                                                 label='Target Object Position [m]')[0]
            handles.append(target_pos_plot)
            Frame.append(target_pos_plot)

            # Plot bounding box for target object if corners are provided
            if corners is not None:
                x, y = corners[:, 1], corners[:, 0]
                target_box_plot = ax_trajectory.plot([x[0], x[1], x[2], x[3], x[0]],
                                                     [y[0], y[1], y[2], y[3], y[0]],
                                                     color='blue', linestyle='-', marker='o', linewidth=4, markersize=16)[0]
                handles.append(target_box_plot)
                Frame.append(target_box_plot)

        # Plot goal position if provided
        if goal_position is not None and goal_yaw is not None:
            goal_pos_plot = ax_trajectory.scatter(goal_position[1], goal_position[0],
                                                color="red", linewidth=4,
                                                label='Subgoal')
            handles.append(goal_pos_plot)
            Frame.append(goal_pos_plot)
            
        # Add legend to trajectory plot
        ax_trajectory.legend(handles=handles)

        return Frame

    @staticmethod
    def save_animation(ani, dest_dir, file_name):
        """
        Saves an animation object to MP4 file in the specified directory.

        Args:
            ani (matplotlib.animation.Animation): The animation object to be saved.
            dest_dir (str): The directory path where the animation files will be saved.
            file_name (str): The base name of the file without extension where the animation will be saved.

        Returns:
            None: This method performs file I/O and does not return any value.
        """
        print("[INFO] Saving animation")
        
        # Define file paths
        gif_file_path = os.path.join(dest_dir, f'{file_name}.gif')
        mp4_file_path = os.path.join(dest_dir, f'{file_name}.mp4')

        # Save to GIF using PillowWriter
        writergif = animation.PillowWriter(fps=10)
        ani.save(gif_file_path, writer=writergif)
        
        # Convert GIF to MP4 and remove GIF file
        clip = mp.VideoFileClip(gif_file_path)
        clip.write_videofile(mp4_file_path)
        os.remove(gif_file_path)
        
        print("[INFO] Animation saved in MP4 format")


@dataclass
class ObjDet:
    """
    Data class for representing object detections.

    Attributes:
        label (str): Label of the detected object.
        label_id (int): Numeric identifier corresponding to the label.
        instance_id (int): Identifier for the detected instance.
        confidence (float): Confidence score of the detection.
        tracking_state (int): State of the object tracking.
        tracking_available (bool): Flag indicating if tracking is available.
        position (List[float]): 3D position of the object (x, y, z).
        position_covariance (List[float]): Covariance of the position estimate.
        velocity (List[float]): Velocity of the object (vx, vy, vz).
        bbox3d (List[List[float]]): 3D bounding box of the object (eight corners).
        bbox2d (List[List[float]]): 2D bounding box of the object (four corners).
    """
    label: str = "None"
    label_id: int = -1
    instance_id: int = -1
    confidence: float = -1.0
    tracking_state: int = -1
    tracking_available: bool = False
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    position_covariance: List[float] = field(default_factory=lambda: [0.0]*6)
    velocity: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    bbox3d: List[List[float]] = field(default_factory=lambda: [[0.0, 0.0, 0.0]]*8)
    bbox2d: List[List[float]] = field(default_factory=lambda: [[0.0, 0.0]]*4)


## msg structure:
## https://github.com/stereolabs/zed-ros-interfaces/tree/main/msg
def object_detection_to_dic(msg: ObjectsStamped)->dict:
    """
    Converts an object detection message to a dictionary format.

    Args:
        msg (ObjectsStamped): The ROS object detection message.

    Returns:
        dict: A dictionary containing the details of the first detected object.
    """
    obj = ObjDet()

    # Check if there is at least one detection
    if len(msg.objects) > 0:
        obj_ros = msg.objects[0]

        # Update ObjDet data class attributes from the ROS message
        obj.label = obj_ros.label
        obj.label_id = obj_ros.label_id
        obj.instance_id = obj_ros.instance_id
        obj.confidence = obj_ros.confidence
        obj.tracking_state = obj_ros.tracking_state
        obj.tracking_available = obj_ros.tracking_available
        obj.position = list(obj_ros.position)
        obj.position_covariance = list(obj_ros.position_covariance)
        obj.velocity = list(obj_ros.velocity)
        obj.bbox3d = bbox_to_list(obj_ros.bounding_box_3d)
        obj.bbox2d = bbox_to_list(obj_ros.bounding_box_2d)

    return {
        "label": obj.label, "label_id": obj.label_id, "instance_id": obj.instance_id,
        "confidence": obj.confidence, "tracking_state": obj.tracking_state, "tracking_available": obj.tracking_available,
        "position": obj.position, "position_covariance": obj.position_covariance, "velocity": obj.velocity,
        "bbox3d": obj.bbox3d, "bbox2d": obj.bbox2d
    }

def bbox_to_list(bbox)->List[float]:
    """
    Converts a bounding box message to a list format.

    Args:
        bbox (BoundingBox): The ROS bounding box message which includes corners as attributes.

    Returns:
        List[List[float]]: A list of points representing the corners of the bounding box.
    """
    corners = []
    for corner in bbox.corners:
        corners.append(list(corner.kp))
    return corners

def get_key_by_value(my_dict, value)->str:
    """
    Finds the first key in a dictionary that corresponds to the given value.

    Args:
        my_dict (dict): The dictionary to search.
        value: The value for which to find the corresponding key.

    Returns:
        The key corresponding to the input value. Returns None if the value is not found.

    """
    
    key = None
    # Iterate through each key-value pair in the dictionary
    for k, v in my_dict.items():
        # Check if the current value matches the input value
        if v == value:
            key = k  # If found, set the key to the current key
            break  # Exit the loop as soon as the key is found
    return key

def odom_to_numpy(msg:Odometry)->np.ndarray:
    """
    Converts an Odometry message to numpy arrays representing position, orientation, linear velocity, and angular velocity.

    Args:
        msg (Odometry): An instance of Odometry.

    Returns:
        A list of numpy arrays: [position, orientation, linear velocity, angular velocity], where each array represents a different aspect of the Odometry message.
    """
    
    # Extract position and orientation from Odometry message
    pos, ori = pose_to_numpy(msg.pose)

    # Extract linear and angular velocity
    lin_vel = [msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z]
    angular_vel = [msg.twist.twist.angular.x,
                msg.twist.twist.angular.y,
                msg.twist.twist.angular.z]

    # Convert the extracted values to numpy arrays and return them
    return [np.array(pos), np.array(ori), np.array(lin_vel), np.array(angular_vel)]


def pose_to_numpy(msg: PoseStamped)->np.ndarray:
    """
    Converts a PoseStamped message to numpy arrays representing position and orientation.

    Args:
        msg (PoseStamped): An instance of PoseStamped.

    Returns:
        A list of numpy arrays: [position, orientation], where each array represents a different aspect of the PoseStamped message.
    """
    
    # Extract position and orientation from PoseStamped message
    pos = [msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z]
    ori = [msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w]

    # Convert the extracted values to numpy arrays and return them
    return [np.array(pos), np.array(ori)]

def np_odom_to_xy_yaw(np_odom:np.ndarray, prev_data, Ao)->dict:
    """
    Extracts x, y coordinates and yaw angle from a numpy representation of Odometry.

    Args:
        np_odom (np.ndarray): Numpy representation of Odometry, typically output of `odom_to_numpy` function.
        t (int): step 
    Returns:
        A dictionary containing 'pos', a list of x and y coordinates, and 'yaw', the yaw angle [radians].
    """
    
    x_gt, y_gt = np_odom[0][0], np_odom[0][1]
    yaw_gt = quat_to_yaw(np_odom[1])
    
    #not in use
    # vt =  np_odom[2][0]
    # wt = np_odom[3][2] if np_odom[3][2] != 0 else 10**-7
    
    if(prev_data is not None):
        pos_in_odom = Ao @ np.array([x_gt, y_gt, 1]).T
        x_in_odom, y_in_odom = pos_in_odom[0], pos_in_odom[1]
        prev_x_in_odom, prev_y_in_odom = prev_data['odom_frame']['position'][0], prev_data['odom_frame']['position'][1]
        dx_in_odom, dy_in_odom = x_in_odom - prev_x_in_odom,y_in_odom - prev_y_in_odom
        
        dyaw = normalize_angle(yaw_gt) - normalize_angle(prev_data['gt_frame']['yaw'])
        prev_yaw_in_odom = prev_data['odom_frame']['yaw']
        yaw_in_odom = prev_yaw_in_odom + dyaw
        yaw_in_odom = normalize_angle(yaw_in_odom)
        
        Ar = get_transform_to_start(prev_x_in_odom,prev_y_in_odom,prev_yaw_in_odom)
        
        pos_rel_to_prev = Ar @ np.array([x_in_odom, y_in_odom, 1]).T
        x_rel, y_rel =  pos_rel_to_prev[0], pos_rel_to_prev[1]
        yaw_rel = dyaw
        
        
    else:
        x_in_odom, y_in_odom = 0. , 0.
        yaw_in_odom = 0.
        dx_in_odom, dy_in_odom = 0. , 0.
        dyaw = 0
        
        x_rel, y_rel =  0. , 0.
        yaw_rel = 0.


    return {'gt_frame':{'position':[x_gt, y_gt], 'yaw': yaw_gt},
            'odom_frame':{'position':[x_in_odom,y_in_odom], 'yaw': yaw_in_odom, 'dpos':[dx_in_odom, dy_in_odom],'dyaw': dyaw},
            'relative_frame':{'position':[x_rel,y_rel], 'yaw': yaw_rel}}

def np_pose_in_odom(np_pose:np.ndarray,Ao)->dict:
    ## TODO: Refactore, add relative goal pose to the current pose of the robot
    x_gt, y_gt = np_pose[0][0], np_pose[0][1]
    yaw_gt = quat_to_yaw(np_pose[1])
    
    pos_in_odom = Ao @ np.array([x_gt, y_gt, 1]).T
    x_in_odom, y_in_odom = pos_in_odom[0], pos_in_odom[1]

    
    return {'gt_frame':{'position':[x_gt, y_gt], 'yaw': yaw_gt},
            'odom_frame':{'position':[x_in_odom,y_in_odom], 'yaw': 0.}}
            
def get_transform_to_start(x_start, y_start, yaw_start)->np.ndarray:
    """
    Computes the inverse transformation matrix from the start position and orientation.

    Calculates the inverse of the homogeneous transformation matrix for the starting coordinates (x_start, y_start)
    and orientation (yaw_start), facilitating transformations to the origin of the coordinate system.

    Args:
        x_start (float): The x-coordinate of the starting position.
        y_start (float): The y-coordinate of the starting position.
        yaw_start (float): The yaw (orientation) in radians at the starting position.

    Returns:
        np.array: The 3x3 inverse transformation matrix.
    """
    # Define the forward transformation matrix based on initial position and yaw
    A = np.array([
        [np.cos(yaw_start), -np.sin(yaw_start), x_start],
        [np.sin(yaw_start),  np.cos(yaw_start), y_start],
        [0, 0, 1]
    ])

    # Calculate the inverse of the transformation matrix
    Ainv = np.linalg.inv(A)
    
    return Ainv


def normalize_angle(angle)->float:
    """
    Normalizes an angle to the range [-π, π].

    Args:
        angle (float): The angle to normalize, in radians.

    Returns:
        float: The normalized angle within [-π, π].
    """
    if -np.pi < angle <= np.pi:
        return angle
    if angle > np.pi:
        angle -= 2 * np.pi
    if angle <= -np.pi:
        angle += 2 * np.pi
    return normalize_angle(angle)

def normalize_angles_array(angles)->np.ndarray:
    """
    Normalizes an array of angles to the range [-π, π].

    Args:
        angles (np.array): An array of angles to normalize, in radians.

    Returns:
        np.array: An array of normalized angles within [-π, π].
    """
    z = np.zeros_like(angles)
    for i in range(angles.shape[0]):
        z[i] = normalize_angle(angles[i])
    return z

def quat_to_yaw(quat: np.ndarray) -> float:
    """
    Converts a quaternion to a yaw angle [radians].

    Args:
        quat (np.ndarray): A numpy array representing a quaternion in the order [x, y, z, w].

    Returns:
        float: The yaw angle derived from the quaternion.
    """
    
    # Unpack the quaternion components
    x, y, z, w = quat[0], quat[1], quat[2], quat[3]
    # Compute the yaw from the quaternion
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return yaw


def image_to_numpy(msg, empty_value=None, output_resolution=None, max_depth=10000, use_bridge=False)->np.ndarray:
    """
    Converts a ROS image message to a numpy array, applying format conversions, depth normalization, and resolution scaling.

    This function supports direct numpy conversions without using CvBridge as well as conversions via CvBridge if specified.

    Args:
        msg (Image): The ROS image message to convert.
        empty_value (float, optional): A value to replace missing or invalid data points in the image. Defaults to None.
        output_resolution (tuple of int, optional): The desired resolution (width, height) for the output image. Defaults to the input image's resolution.
        max_depth (int): The maximum depth used for normalizing depth images. Only applies to depth images. Defaults to 5000.
        use_bridge (bool): Set to True to use CvBridge for converting image messages. Defaults to False.

    Returns:
        np.ndarray: The converted image as a numpy array.
    """
    
    # Set default output resolution to the input message's resolution if not specified
    if output_resolution is None:
        output_resolution = (msg.width, msg.height)

    # Determine the type of image encoding
    is_rgb = "8" in msg.encoding.lower()
    is_depth16 = "16" in msg.encoding.lower()
    is_depth32 = "32" in msg.encoding.lower()

    if not use_bridge:
        # Convert the ROS image to a numpy array directly
        if is_rgb:
            data = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)[:,:,:3].copy()
        elif is_depth16:
            data = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width).copy()
            max_depth_clip = max_depth if max_depth else np.max(data)
            # data = 1 - (np.clip(data, a_min=0, a_max=max_depth_clip) / max_depth_clip)
            data = np.clip(data, a_min=0, a_max=max_depth_clip) / max_depth_clip

            data = np.array(255*data.astype(np.float32),dtype=np.uint8)
        elif is_depth32:
            data = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width).copy()
    else:
        # Use CvBridge to convert the ROS image
        bridge = CvBridge()
        try:
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            if is_rgb:
                data = cv_img[:,:,:3]
            elif is_depth16 or is_depth32:
                data = np.clip(cv_img, 0, max_depth) / max_depth
                data = np.uint8(data * 255)
        except CvBridgeError as e:
            print(f"Error converting image: {e}")
            return None

    # Replace specified 'empty values' or NaNs with a fixed value
    if empty_value:
        mask = np.isclose(abs(data), empty_value)
    else:
        mask = np.isnan(data)
    
    fill_value = np.percentile(data[~mask], 99)
    data[mask] = fill_value

    # Resize the image if a specific output resolution is set
    if output_resolution != (msg.width, msg.height):
        data = cv2.resize(data, dsize=(output_resolution[0], output_resolution[1]), interpolation=cv2.INTER_AREA)

    return data

    ### 32bit depth image 
    #### NOT WORKING YET
    # Create a mask to filter out NaN and Inf values
    # valid_mask = np.isfinite(depth_image)

    # # Replace NaN and Inf values with a value outside the desired range (e.g., 0)
    # depth_image[~valid_mask] = 0

    # # Normalize the depth values to the range [0, 255]
    # min_depth = np.min(depth_image[valid_mask])
    # max_depth = np.max(depth_image[valid_mask])
    # normalized_depth = ((depth_image - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)

    # # Convert the normalized depth image to an 8-bit unsigned integer image
    # depth_8bit = cv2.convertScaleAbs(normalized_depth)

##### NOT WORKING YET #####
def image_compressed_to_numpy(msg:CompressedImage)->np.ndarray:
    """
    Converts a ROS CompressedImage message to a numpy array.

    Args:
        msg (CompressedImage): A ROS CompressedImage message.

    Returns:
        np.ndarray: A numpy array representing the image.
    """
    
    # Convert CompressedImage to OpenCV Image using np.frombuffer
    np_img = np.frombuffer(msg.data, dtype=np.uint8)
    np_img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    return np_img

def reset_processed_bags(file_path):
    # Load the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Reset 'processed_bags' to an empty list
    data['processed_bags'] = []
    
    # Save the modified data back to the file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

    print(f"Processed bags reset in {file_path}")
