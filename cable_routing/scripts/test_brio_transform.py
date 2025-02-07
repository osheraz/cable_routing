import sys
import math
import numpy as np
import cv2
import time
from autolab_core import RigidTransform, Point, CameraIntrinsics
from cable_routing.env.robots.yumi import YuMiRobotEnv
from cable_routing.configs.envconfig import ExperimentConfig
import tyro
from sensor_msgs.msg import CameraInfo
from cable_routing.env.ext_camera.brio_camera import BRIOSensor

# Constants
TABLE_HEIGHT = 0.0582
SCALE_FACTOR = 0.25  # Display scaling factor
ZOOM_SIZE = 100  # Zoom-in region size

def get_world_coord_from_pixel_coord(pixel_coord, cam_intrinsics, cam_extrinsics):
    """
    Convert pixel coordinates to world coordinates using camera intrinsics and extrinsics.
    """
    pixel_coord = np.array(pixel_coord)
    point_3d_cam = np.linalg.inv(cam_intrinsics._K).dot(np.r_[pixel_coord, 1.04 - TABLE_HEIGHT])
    point_3d_world = cam_extrinsics.matrix.dot(np.r_[point_3d_cam, 1.0])
    point_3d_world = point_3d_world[:3] / point_3d_world[3]
    point_3d_world[-1] = TABLE_HEIGHT
    return point_3d_world

def click_event(event, x, y, flags, param):
    """
    Handles mouse click events on the resized image.
    Maps the clicked coordinates to the original image and stores them for later processing.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        # Convert coordinates back to the original resolution
        orig_x = int(x / SCALE_FACTOR)
        orig_y = int(y / SCALE_FACTOR)
        param["u"], param["v"] = orig_x, orig_y

        # Display a zoomed-in view
        original_frame = param["original_frame"]
        half_zoom = ZOOM_SIZE // 2
        roi = original_frame[max(0, orig_y - half_zoom):orig_y + half_zoom,
                             max(0, orig_x - half_zoom):orig_x + half_zoom]

        # if roi.shape[0] > 0 and roi.shape[1] > 0:
        #     zoomed_roi = cv2.resize(roi, (ZOOM_SIZE * 4, ZOOM_SIZE * 4), interpolation=cv2.INTER_LINEAR)
        #     cv2.imshow("Zoomed In View", zoomed_roi)

def main(args: ExperimentConfig):
    """
    Main function to run the robot-camera interaction loop.
    """

    # Initialize YuMi robot
    yumi = YuMiRobotEnv(args.robot_cfg)
    yumi.close_grippers()
    yumi.move_to_home()

    # Initialize BRIO camera
    brio_cam = BRIOSensor(device=0)

    # Load camera extrinsics
    T_CAM_BASE = RigidTransform.load("/home/osheraz/cable_routing/data/zed/brio_to_world.tf").as_frames(
        from_frame="brio", to_frame="base_link"
    )
    print("Camera Extrinsics:\n", T_CAM_BASE)

    # Load camera intrinsics
    CAM_INTR = CameraIntrinsics(fx=args.camera_cfg.fx,
                                fy=args.camera_cfg.fy,
                                cx=args.camera_cfg.cx,
                                cy=args.camera_cfg.cy,
                                width=args.camera_cfg.width,
                                height=args.camera_cfg.height,
                                frame=args.camera_cfg.frame)

    # Capture an image
    frame = brio_cam.read()

    # Resize the image for display
    resized_frame = cv2.resize(frame, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR, interpolation=cv2.INTER_AREA)

    # Display the resized image
    params = {"original_frame": frame, "u": None, "v": None}
    cv2.imshow("Resized Image", resized_frame)
    cv2.setMouseCallback("Resized Image", click_event, param=params)

    while params["u"] is None or params["v"] is None:
        if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC key
            cv2.destroyAllWindows()
            return

    # Compute the world coordinates from the selected pixel
    pixel_coord = [params["u"], params["v"]]
    world_coord = get_world_coord_from_pixel_coord(pixel_coord, CAM_INTR, T_CAM_BASE)
    print("World Coordinate: ", world_coord)

    # Move the robot end-effector to the target point
    target_pose = RigidTransform(rotation=RigidTransform.x_axis_rotation(math.pi), translation=world_coord)
    target_pose.translation[2] += 0.2  # Move above the point
    yumi.set_ee_pose(left_pose=target_pose)

    target_pose.translation[2] -= 0.1  # Move down to the point
    yumi.set_ee_pose(left_pose=target_pose)

    input("Press Enter to continue...")
    yumi.set_ee_pose(left_pose=yumi.get_ee_pose()[0])  # Move back to original pose

    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = tyro.cli(ExperimentConfig)
    main(args)
