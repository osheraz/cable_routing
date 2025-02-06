import sys
import math
import os
import yaml
import numpy as np
import cv2
import time
from autolab_core import RigidTransform, Point
from tqdm import tqdm
from cable_routing.env.robots.yumi import YuMiRobotEnv
from cable_routing.env.ext_camera.zed_camera_pyzed import Zed
from cable_routing.configs.envconfig import ZedMiniConfig
from cable_routing.configs.envconfig import ExperimentConfig

import pyzed.sl as sl
import tyro

# TODO make relative.
world_to_extrinsic_zed_path = '/home/osheraz/cable_routing/data/zed/zed2world.tf'
world_to_extrinsic_zed = RigidTransform.load(world_to_extrinsic_zed_path)


def setup_zed_camera(camera_parameters):
    """ Sets up the ZED camera with the given parameters. """
    zed = Zed(
        flip_mode=camera_parameters.flip_mode,
        resolution=camera_parameters.resolution,
        fps=camera_parameters.fps,
        # cam_id=camera_parameters.id
    )

    zed.cam.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, camera_parameters.gain)
    zed.cam.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, camera_parameters.exposure)
    return zed


def click_event(event, u, v, flags, param):
    """ Handles mouse click events to get pixel coordinates. """
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel_value = param["img"][v, u]
        print(f"Pixel coordinates: (u={u}, v={v}) - Pixel value: {pixel_value}")

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(param["img"], f"({u},{v})", (u, v), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(param["img"], str(pixel_value), (u, v + 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        time.sleep(0.5)
        param["u"] = u
        param["v"] = v


def main(args: ExperimentConfig):
    """ Main function to run the robot-camera interaction loop. """
    
    # Initialize YuMi robot
    yumi = YuMiRobotEnv(args.robot_cfg)
    yumi.close_grippers()
    yumi.move_to_home()

    # Load camera configuration
    camera_parameters = ZedMiniConfig()
    zed = setup_zed_camera(camera_parameters)

    # Retrieve camera calibration parameters
    calibration_params = zed.cam.get_camera_information().camera_configuration.calibration_parameters
    f_x, f_y = calibration_params.left_cam.fx, calibration_params.left_cam.fy
    c_x, c_y = calibration_params.left_cam.cx, calibration_params.left_cam.cy

    for _ in tqdm(range(3)):
        # Get the current end-effector pose
        left_pose, _ = yumi.get_ee_pose()
        previous_pose = left_pose  # Store for later use

        # Get RGB and depth images
        left_img, depth = zed.get_rgb_depth()
        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)

        # Display image and wait for user selection
        params = {"img": left_img.copy(), "u": None, "v": None}
        cv2.imshow("Image", left_img)
        cv2.setMouseCallback("Image", click_event, param=params)

        while params["u"] is None or params["v"] is None:
            if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC key
                break

        # Get user-selected pixel coordinates
        u, v = params["u"], params["v"]
        Z = depth[v, u]
        X = ((u - c_x) * Z) / f_x
        Y = ((v - c_y) * Z) / f_y
        print(f"Projected point: X={X}, Y={Y}, Z={Z}")

        # Transform point from camera to robot frame
        point = Point(np.array([X, Y, Z]), frame="zed_extrinsic")
        point_in_robot = world_to_extrinsic_zed * point
        print("Point in robot frame:", point_in_robot.data)

        # Move the robot end-effector to the target point
        # target_pose = RigidTransform(rotation=RigidTransform.x_axis_rotation(math.pi), translation=point_in_robot.data)
        # target_pose.translation[2] += 0.2  # Move above the point
        # yumi.set_ee_pose(left_pose=target_pose)

        # target_pose.translation[2] -= 0.2  # Move down to the point
        # yumi.set_ee_pose(left_pose=target_pose)

        # input("Press Enter to continue...")
        # yumi.set_ee_pose(left_pose=previous_pose)  # Move back to original pose

    # Close the camera
    zed.close()


if __name__ == "__main__":

    args = tyro.cli(ExperimentConfig)

    main(args)
