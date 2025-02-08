import rospy
from sensor_msgs.msg import Image
from cable_routing.env.ext_camera.ros.utils.image_utils import image_msg_to_numpy
import numpy as np
import cv2


class ZedCameraSubscriber:

    def __init__(self,
                  topic_depth='/zedm/zed_node/depth/depth_registered',
                  topic_rgb='/zedm/zed_node/rgb/image_rect_color',
                  display=False):
        
        self.w = 640  
        self.h = 360 
        self.cam_type = 'd'
        self.far_clip = 1
        self.near_clip = 0.1
        self.dis_noise = 0.00
        
        self.display = display
        self.zed_init = False
        self.init_success = False

        self._topic_name = topic_depth
        self._rgb_name = topic_rgb

        rospy.loginfo("(topic_name) Subscribing to Images to topic  %s", self._topic_name)
        self._check_rgb_ready()
        self._check_depth_ready()

        self._depth_subscriber = rospy.Subscriber(self._topic_name, Image, self.image_callback, queue_size=2)
        self._image_subscriber = rospy.Subscriber(self._rgb_name, Image, self.rgb_callback, queue_size=2)

    def _check_rgb_ready(self):

        self.raw_frame = None
        rospy.logdebug(
            "Waiting for '{}' to be READY...".format(self._rgb_name))
        while self.raw_frame is None and not rospy.is_shutdown():
            try:
                self.raw_frame = rospy.wait_for_message(
                    '{}'.format(self._rgb_name), Image, timeout=5.0)
                rospy.logdebug(
                    "Current '{}' READY=>".format(self._rgb_name))
                self.start_time = rospy.get_time()
                self.raw_frame = image_msg_to_numpy(self.raw_frame)

            except:
                rospy.logerr(
                    "Current '{}' not ready yet, retrying for getting image".format(self._rgb_name))
                
        return self.raw_frame
    
    def _check_depth_ready(self):
        
        self.last_frame = None
        rospy.logdebug("Waiting for '{}' to be READY...".format(self._topic_name))

        while self.last_frame is None and not rospy.is_shutdown():
            try:
                self.last_frame = rospy.wait_for_message(
                    '{}'.format(self._topic_name), Image, timeout=5.0)
                rospy.logdebug("Current '{}' READY=>".format(self._topic_name))
                self.zed_init = True
                self.last_frame = image_msg_to_numpy(self.last_frame)
                self.last_frame = np.expand_dims(self.last_frame, axis=0)
                self.start_time = rospy.get_time()
            except:
                rospy.logerr("Current '{}' not ready yet, retrying...".format(self._topic_name))
        return self.last_frame


    def rgb_callback(self, msg):
        try:
            self.raw_frame = image_msg_to_numpy(msg)
        except Exception as e:
            print(e)
            return
        
    def image_callback(self, msg):
        try:
            frame = image_msg_to_numpy(msg)
        except Exception as e:
            print(e)
        else:
            frame = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_AREA)
            frame = np.expand_dims(frame, axis=0)
            self.last_frame = self.process_depth_image(frame)


    def get_frames(self):
        depth = self.last_frame
        rgb = self.raw_frame
        return rgb, depth
    
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
    rospy.init_node('zed_pub')

    zed_cam = ZedCameraSubscriber(display=True) 
    rate = rospy.Rate(60)

    while not rospy.is_shutdown():
        if zed_cam.display and zed_cam.last_frame is not None and zed_cam.raw_frame is not None:

            cv2.imshow("Depth Image", zed_cam.last_frame.transpose(1, 2, 0))
            cv2.imshow("RGB Image", zed_cam.raw_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break 

        rate.sleep()
