import rospy
from cable_routing.env.ext_camera.ros.utils.image_utils import image_msg_to_numpy
import numpy as np
import cv2
from sensor_msgs.msg import CameraInfo, Image
from autolab_core import RigidTransform, Point, CameraIntrinsics


class ZedCameraSubscriber:

    def __init__(
        self,
        topic_depth="/zedm/zed_node/depth/depth_registered",
        topic_rgb="/zedm/zed_node/rgb/image_rect_color",
        display=False,
    ):

        camera_info = rospy.wait_for_message(
            "/zedm/zed_node/depth/camera_info", CameraInfo
        )

        self.intrinsic = CameraIntrinsics(
            fx=camera_info.K[0],
            fy=camera_info.K[4],
            cx=camera_info.K[2],
            cy=camera_info.K[5],
            width=camera_info.width,
            height=camera_info.height,
            frame="zed",
        )

        self.w = camera_info.width
        self.h = camera_info.height

        # TODO from cfg
        self.far_clip = 1
        self.near_clip = 0.1
        self.dis_noise = 0.00

        self.display = display
        self.zed_init = False
        self.init_success = False

        self._topic_name = topic_depth
        self._rgb_name = topic_rgb

        rospy.loginfo(
            "(topic_name) Subscribing to Images to topic  %s", self._topic_name
        )
        self._check_rgb_ready()
        self._check_depth_ready()

        # TODO add message filters to sync
        self._depth_subscriber = rospy.Subscriber(
            self._topic_name, Image, self.depth_callback, queue_size=2
        )
        self._image_subscriber = rospy.Subscriber(
            self._rgb_name, Image, self.rgb_callback, queue_size=2
        )

    def _check_rgb_ready(self):

        self.rgb_image = None
        rospy.loginfo("Waiting for '{}' to be READY...".format(self._rgb_name))
        while self.rgb_image is None and not rospy.is_shutdown():
            try:
                self.rgb_image = rospy.wait_for_message(
                    "{}".format(self._rgb_name), Image, timeout=5.0
                )
                rospy.loginfo("Current '{}' READY=>".format(self._rgb_name))
                self.start_time = rospy.get_time()
                self.rgb_image = image_msg_to_numpy(self.rgb_image)

            except:
                rospy.logerr(
                    "Current '{}' not ready yet, retrying for getting image".format(
                        self._rgb_name
                    )
                )

        return self.rgb_image

    def _check_depth_ready(self):

        self.depth_image = None
        rospy.loginfo("Waiting for '{}' to be READY...".format(self._topic_name))

        while self.depth_image is None and not rospy.is_shutdown():
            try:
                self.depth_image = rospy.wait_for_message(
                    "{}".format(self._topic_name), Image, timeout=5.0
                )
                rospy.loginfo("Current '{}' READY=>".format(self._topic_name))
                self.zed_init = True
                self.depth_image = image_msg_to_numpy(self.depth_image)
                self.start_time = rospy.get_time()
            except:
                rospy.logerr(
                    "Current '{}' not ready yet, retrying...".format(self._topic_name)
                )
        return self.depth_image

    def rgb_callback(self, msg):
        try:
            self.rgb_image = image_msg_to_numpy(msg)
        except Exception as e:
            print(e)
            return

    def depth_callback(self, msg):
        try:
            self.depth_image = image_msg_to_numpy(msg)
        except Exception as e:
            rospy.logerr(f"Depth callback error: {e}")

    def get_frames(self):
        depth = self.depth_image
        rgb = self.rgb_image
        return rgb, depth

    def get_rgb(self):

        return self.rgb_image

    def process_depth_image(self, depth_image):

        # depth_image = self.crop_depth_image(depth_image)
        depth_image = np.clip(depth_image, self.near_clip, self.far_clip)
        depth_image = self.normalize_depth_image(depth_image)
        return depth_image

    def normalize_depth_image(self, depth_image):
        depth_image = depth_image
        depth_image = (depth_image - self.near_clip) / (self.far_clip - self.near_clip)
        return depth_image

    def crop_depth_image(self, depth_image):
        return depth_image


if __name__ == "__main__":
    rospy.init_node("zed_pub")

    zed_cam = ZedCameraSubscriber(display=True)
    rate = rospy.Rate(60)

    while not rospy.is_shutdown():
        if (
            zed_cam.display
            and zed_cam.depth_image is not None
            and zed_cam.rgb_image is not None
        ):

            cv2.imshow(
                "Depth Image", np.expand_dims(zed_cam.depth_image, 0).transpose(1, 2, 0)
            )
            cv2.imshow("RGB Image", zed_cam.rgb_image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        rate.sleep()
