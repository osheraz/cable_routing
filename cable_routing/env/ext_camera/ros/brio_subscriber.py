import rospy
from sensor_msgs.msg import Image
from cable_routing.env.ext_camera.ros.image_utils import image_msg_to_numpy
import threading


class BRIOSubscriber:
    def __init__(self, topic_name="/camera/camera_0/image_raw", queue_size=1):
        """
        Initialize the BRIOSubscriber.

        Parameters:
        - topic_name: The name of the ROS topic to subscribe to.
        - queue_size: The size of the message queue for the subscriber.
        """
        self.topic_name = topic_name
        self.latest_frame = None

        # Initialize the ROS node if it hasn't been initialized yet
        if not rospy.core.is_initialized():
            rospy.init_node("brio_subscriber", anonymous=True)

        self._check_rgb_ready()
        # Create the image subscriber
        self.image_sub = rospy.Subscriber(
            self.topic_name, Image, self.callback, queue_size=queue_size
        )
        rospy.loginfo(f"Subscribed to {self.topic_name}")

    def _check_rgb_ready(self):

        self.latest_frame = None
        rospy.logdebug("Waiting for '{}' to be READY...".format(self.topic_name))

        while self.latest_frame is None and not rospy.is_shutdown():
            try:
                self.latest_frame = rospy.wait_for_message(
                    "{}".format(self.topic_name), Image, timeout=5.0
                )
                rospy.logdebug("Current '{}' READY=>".format(self.topic_name))
                self.start_time = rospy.get_time()
                self.latest_frame = image_msg_to_numpy(self.latest_frame)

            except:
                rospy.logerr(
                    "Current '{}' not ready yet, retrying for getting image".format(
                        self.topic_name
                    )
                )

        return self.latest_frame

    def callback(self, msg):
        """
        Callback function that gets executed upon receiving an image message.

        Parameters:
        - msg: The ROS Image message.
        """
        try:
            cv_image = image_msg_to_numpy(msg)
            self.latest_frame = cv_image
        except Exception as e:
            rospy.logerr(f"Failed to convert image: {str(e)}")

    def get_frame(self):
        """
        Retrieve the latest received frame.

        Returns:
        """
        return self.latest_frame.copy()


if __name__ == "__main__":
    try:
        brio_subscriber = BRIOSubscriber()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
