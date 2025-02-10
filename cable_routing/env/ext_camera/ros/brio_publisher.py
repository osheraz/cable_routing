import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import threading
from cable_routing.env.ext_camera.brio_camera import BRIOSensor


class CameraPublisher:
    def __init__(self, device_id=0, name="camera_0", fps=10, init_node=False):
        if init_node:
            rospy.init_node("camera_publisher", anonymous=True)

        self.name = name
        self.running = False
        self.thread = None
        self.bridge = CvBridge()
        self.fps = fps

        # Initialize the BRIO camera
        self.camera = BRIOSensor(device=device_id)
        self.width = self.camera.width
        self.height = self.camera.height
        if not self.camera.is_running:
            rospy.logerr("Failed to initialize BRIO camera!")
            return

        # Create the image publisher
        self.image_pub = rospy.Publisher(
            f"/camera/{self.name}/image_raw", Image, queue_size=1
        )
        self.rate = rospy.Rate(self.fps)

        rospy.loginfo(f"Started camera publisher {self.name} at {self.fps} FPS")

    def _run(self):
        """Internal run method that runs in the thread"""
        while not rospy.is_shutdown() and self.running:
            frame = self.camera.read()
            if frame is not None:
                try:
                    ros_image = self.bridge.cv2_to_imgmsg(frame, "rgb8")
                    ros_image.header.stamp = rospy.Time.now()
                    ros_image.header.frame_id = self.name
                    self.image_pub.publish(ros_image)
                except Exception as e:
                    rospy.logerr(f"Failed to publish image from {self.name}: {str(e)}")
            self.rate.sleep()

    def get_frame(self):

        return self.camera.read()

    def start(self):
        """Start the camera publisher in a new thread"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run)
            self.thread.start()
            rospy.loginfo(f"Camera {self.name} thread started")

    def stop(self):
        """Stop the camera publisher thread"""
        self.running = False
        if self.thread is not None:
            self.thread.join()
            self.thread = None
            rospy.loginfo(f"Camera {self.name} thread stopped")

    def __del__(self):
        """Cleanup camera resources"""
        self.stop()
        if hasattr(self.camera, "stop"):
            self.camera.stop()


if __name__ == "__main__":
    try:
        rospy.init_node("multi_camera_publisher")
        camera0 = CameraPublisher(device_id=0, name="camera_0")
        camera0.start()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        if "camera0" in locals():
            camera0.stop()
