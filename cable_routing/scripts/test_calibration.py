import cv2
import numpy as np
from autolab_core import RigidTransform, Point, CameraIntrinsics
import os
import tyro
from yumi_jacobi.interface import Interface
import time
from raftstereo.zed_stereo import Zed


def main(is_zed: bool):
    cam_name = "zed" if is_zed else "brio"

    script_dir = os.path.dirname(os.path.realpath(__file__))
    config_dir = os.path.join(script_dir, os.pardir, f"camera_info/{cam_name}")
    save_dir = os.path.join(config_dir, "test_calibration")
    os.makedirs(save_dir, exist_ok=True)
    camera_to_world = RigidTransform.load(os.path.join(config_dir, f"{cam_name}2world.tf"))
    
    if is_zed:
        zed = Zed(flip_mode=False, cam_id=None, is_res_1080=False)
    else:
        camera_thread = CameraThread(camera_index=0, fps=30)
        camera_thread.start()
    # camera_thread.change_focus(60)
    
    if is_zed:
        camera_matrix = zed.get_K()
        dist_coeffs = zed.get_dist_coeffs()
    else:
        intrinsics = np.load(os.path.join(config_dir, "camera_calibration.npz"))
        camera_matrix = intrinsics["camera_matrix"]
        dist_coeffs = intrinsics["dist_coeffs"]


    interface = Interface(
        speed=0.26,
    )

    interface.home()
    interface.calibrate_grippers()
    interface.close_grippers()

    

    wp1_l = RigidTransform(
        rotation=[[-1, 0, 0], [0, 1, 0], [0, 0, -1]], translation=[0.4, 0.1, 0.2]
    )
    wp2_l = RigidTransform(
        rotation=[[-1, 0, 0], [0, 1, 0], [0, 0, -1]], translation=[0.4, 0.1, 0.3]
    )
    wp3_l = RigidTransform(
        rotation=[[-1, 0, 0], [0, 1, 0], [0, 0, -1]], translation=[0.5, 0.1, 0.2]
    )

    cfgs = []
    wps = []
    for i in range(9):
        if i < 3:
            wp1_l.translation[1] -= 0.05
            curr_wp = wp1_l
        elif i < 6:
            wp2_l.translation[1] -= 0.05
            curr_wp = wp2_l
        elif i < 9:
            wp3_l.translation[1] -= 0.05
            curr_wp = wp3_l

        interface.go_linear_single(r_target=curr_wp)
        time.sleep(1.0)

        wps.append(curr_wp.copy())

        # r_joints = interface.get_joint_positions("right")
        # r_gripper = interface.driver_right.get_gripper_pos().copy()
        # l_joints = interface.get_joint_positions("left")
        # l_gripper = interface.driver_left.get_gripper_pos().copy()
        # curr_cfg = [r_joints, r_gripper, l_joints, l_gripper]
        # cfgs.append(curr_cfg.copy())

        gripper_in_world = interface.get_FK("right").translation
        world_to_camera = camera_to_world.inverse()

        # TODO: wrap this in a function to test without needing the robot code
        gripper_in_camera = world_to_camera * Point(gripper_in_world, frame="world")
        gripper_point = np.array(
            [[gripper_in_camera.x, gripper_in_camera.y, gripper_in_camera.z]]
        )  # 3D point as a 1x3 array
        gripper_pixel, _ = cv2.projectPoints(
            gripper_point,
            np.zeros((3, 1)),
            np.zeros((3, 1)),
            camera_matrix,
            dist_coeffs,
        )

        print(gripper_pixel)

        if is_zed:
            img_l, _ = zed.get_rgb()
            image = img_l 
        else:
            image = camera_thread.get_frame()
        image_with_point = cv2.circle(
            image,
            (int(gripper_pixel[0][0][0]), int(gripper_pixel[0][0][1])),
            radius=15,
            color=(0, 0, 255),
            thickness=-1,
        )

        cv2.imwrite(f"{save_dir}/frame_{i:04d}.jpg", image_with_point)

    # np.save(
    #     os.path.join(save_dir, "wps.npy"),
    #     tested_wps,
    # )

    # np.save(
    #     os.path.join(save_dir, "cfgs.npy"),
    #     tested_cfgs,
    # )

    if not is_zed:
        camera_thread.stop()
    cv2.destroyAllWindows()
    exit()


if __name__ == "__main__":
    tyro.cli(main)