import rospy
import numpy as np
import cv2
from tqdm import tqdm
from cable_routing.env.robots.yumi import YuMiRobotEnv
import tyro
from autolab_core import RigidTransform, Point, CameraIntrinsics
from cable_routing.configs.envconfig import ExperimentConfig
from cable_routing.env.ext_camera.ros.zed_camera import ZedCameraSubscriber
from cable_routing.handloom.handloom_pipeline.single_tracer import CableTracer
from cable_routing.env.board.board import Board
from cable_routing.env.ext_camera.utils.img_utils import (
    crop_img,
    crop_board,
    select_target_point,
    get_world_coord_from_pixel_coord,
    pick_target_on_path,
    find_nearest_point,
)  # Split to env_utils, img_utils etc..


class ExperimentEnv:
    """Superclass for all Robots environments."""

    def __init__(
        self,
        exp_config: ExperimentConfig,
    ):

        rospy.logwarn("Setting up the environment")

        self.robot = YuMiRobotEnv(exp_config.robot_cfg)
        rospy.sleep(2)

        # todo: add cfg support for this 2 to
        self.zed_cam = ZedCameraSubscriber()
        while self.zed_cam.rgb_image is None or self.zed_cam.depth_image is None:
            rospy.sleep(0.1)
            rospy.loginfo("Waiting for images from ZED camera...")

        self.T_CAM_BASE = RigidTransform.load(
            "/home/osheraz/cable_routing/data/zed/zed_to_world.tf"
        ).as_frames(from_frame="zed", to_frame="base_link")

        self.tracer = CableTracer()
        self.set_board_region()  # TODO: find a better way..
        # TODO: add support for this in cfg
        # TODO: modify to Jaimyn code
        self.board = Board(
            config_path="/home/osheraz/cable_routing/data/board_config.json"
        )

        rospy.logwarn("Env is ready")

    def set_board_region(self, img=None):

        if img == None:
            img = self.zed_cam.rgb_image
        _, (self.point1, self.point2) = crop_board(img)

    def check_calibration(self):
        """we are going to poke all the clips/plugs etc"""

        # TODO - above board
        self.robot.move_to_home()

        self.board.visualize_board(self.zed_cam.rgb_image)

        clips = self.board.get_clips()

        abort = False
        for clip in clips:

            if not abort:
                clip_type = clip["type"]
                clip_ori = clip["orientation"]
                pixel_coord = (clip["x"], clip["y"])
                world_coord = get_world_coord_from_pixel_coord(
                    pixel_coord, self.zed_cam.camera_info, self.T_CAM_BASE
                )

                print(
                    f"Poking clip {clip_type}, orientation: {clip_ori} at: {world_coord}"
                )

                # TODO add support for ori
                self.robot.single_hand_grasp(world_coord, slow_mode=True)
                abort = input("Abort? (y/n): ") == "y"

    def update_cable_path(self, end_point=None):

        path, _ = self.trace_cable(end_point=end_point)

        self.board.set_cable_path(path)

    def convert_path_to_world_coord(self, path):

        world_path = []
        for pixel_coord in path:
            world_coord = get_world_coord_from_pixel_coord(
                pixel_coord, self.zed_cam.camera_info, self.T_CAM_BASE
            )
            world_path.append(world_coord)

        return world_path

    def goto_cable_node(self, path):

        frame = self.zed_cam.get_rgb()

        move_to_pixel = pick_target_on_path(frame, path)

        move_to_pixel, idx = find_nearest_point(path, move_to_pixel)

        world_coord = get_world_coord_from_pixel_coord(
            move_to_pixel, self.zed_cam.camera_info, self.T_CAM_BASE
        )

        path_in_world = self.convert_path_to_world_coord(path)

        # TODO:find nearest grasp point on the cable
        # TODO:include orientation
        # TODO:check collision

        path = self.board.get_cable_path()

        self.robot.single_hand_grasp(world_coord, slow_mode=True)

    def get_extrinsic(self):

        return self.T_CAM_BASE

    def trace_cable(self, img=None, end_point=None):

        if img == None:
            img = self.zed_cam.rgb_image

        # TODO: find a better way to set the board region, Handloom related
        img = crop_img(img, self.point1, self.point2)

        if end_point == None:
            end_point = select_target_point(img)

        path, status = self.tracer.trace(img=img, endpoints=end_point)
        cv2.destroyAllWindows()

        print("Tracing status:", status)

        path = [(x + self.point1[0], y + self.point2[0]) for x, y in path]

        return path, status

    #####################################
    #####################################

    def get_obs(self):

        obs = {}
        if self.with_arm:
            ft = self.arm.robotiq_wrench_filtered_state.tolist()
            pos, quat = self.arm.get_ee_pose()
            joints = self.arm.get_joint_values()

            obs.update({"joints": joints, "ee_pose": pos + quat, "ft": ft})

        if self.with_tactile:
            left, right, bottom = self.tactile.get_frames()
            obs["frames"] = (left, right, bottom)

        if self.with_depth:
            img, seg = self.zed.get_frame()
            obs["img"] = img
            obs["seg"] = seg

        if self.with_pcl:
            pcl, sync_rgb, sync_depth, sync_seg, sync_rec_rgb = self.pcl_gen.get_pcl()
            obs["pcl"] = pcl
            obs["rgb"] = sync_rec_rgb
            obs["img"] = sync_depth
            obs["seg"] = sync_seg

        return obs

    def align_and_grasp(
        self,
    ):

        # TODO change align and grasp to dof_relative funcs without moveit

        for i in range(5):

            # ee_pos, ee_quat = self.arm.get_ee_pose()
            ee_pose = self.arm.move_manipulator.get_cartesian_pose_moveit()
            ee_pos = [ee_pose.position.x, ee_pose.position.y, ee_pose.position.z]
            ee_quat = [
                ee_pose.orientation.x,
                ee_pose.orientation.y,
                ee_pose.orientation.z,
                ee_pose.orientation.w,
            ]
            obj_pos = self.tracker.get_obj_pos()
            obj_pos[-1] += 0.075

            if not np.isnan(np.sum(obj_pos)):

                # added delta_x/delta_y to approximately center the object
                ee_pos[0] = obj_pos[0] - 0.02
                ee_pos[1] = obj_pos[1] - 0.01
                ee_pos[2] = obj_pos[2] - 0.01

                # Orientation is different due to moveit orientation, kinova/orientation ( -0.707,0.707,0,0 ~ 0.707,-0.707,0,0)
                ee_target = geometry_msgs.msg.Pose()
                ee_target.orientation.x = ee_quat[0]
                ee_target.orientation.y = ee_quat[1]
                ee_target.orientation.z = ee_quat[2]
                ee_target.orientation.w = ee_quat[3]

                ee_target.position.x = ee_pos[0]
                ee_target.position.y = ee_pos[1]
                ee_target.position.z = ee_pos[2]
                self.arm_movement_result = self.arm.set_ee_pose(ee_target)

                self.grasp()

                return True
            else:
                rospy.logerr("Object is undetectable, attempt: " + str(i))

        return False

    def align_and_release(self, init_plug_pose):

        # TODO change align and grasp to dof_relative funcs without moveit

        for i in range(5):

            ee_pose = self.arm.move_manipulator.get_cartesian_pose_moveit()
            ee_pos = [ee_pose.position.x, ee_pose.position.y, ee_pose.position.z]
            ee_quat = [
                ee_pose.orientation.x,
                ee_pose.orientation.y,
                ee_pose.orientation.z,
                ee_pose.orientation.w,
            ]
            obj_pos = self.tracker.get_obj_pos()
            obj_height = 0
            init_delta_height = 0.03

            if not np.isnan(np.sum(obj_pos)):

                # added delta_x/delta_y to approximately center the object
                ee_pos[0] = init_plug_pose[0] + (ee_pos[0] - obj_pos[0])
                ee_pos[1] = init_plug_pose[1] + (ee_pos[1] - obj_pos[1])
                ee_pos[2] = (
                    init_plug_pose[2] + obj_pos[2] - obj_height + init_delta_height
                )

                # Orientation is different due to moveit orientation,
                # kinova/orientation ( -0.707,0.707,0,0 ~ 0.707,-0.707,0,0)

                ee_target = geometry_msgs.msg.Pose()
                ee_target.orientation.x = ee_quat[0]
                ee_target.orientation.y = ee_quat[1]
                ee_target.orientation.z = ee_quat[2]
                ee_target.orientation.w = ee_quat[3]

                ee_target.position.x = ee_pos[0]
                ee_target.position.y = ee_pos[1]
                ee_target.position.z = ee_pos[2]
                self.arm_movement_result = self.arm.set_ee_pose(ee_target)

                self.release()

                return True
            else:
                rospy.logerr("Object is undetectable, attempt: " + str(i))

        return False

    def get_info_for_control(self):
        pos, quat = self.arm.get_ee_pose()

        joints = self.arm.get_joint_values()
        jacob = self.arm.get_jacobian_matrix().tolist()

        return {
            "joints": joints,
            "ee_pose": pos + quat,
            "jacob": jacob,
        }
