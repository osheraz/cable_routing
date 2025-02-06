import open3d as o3d
import numpy as np
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cable_routing.env.ext_camera.ros.image_utils import image_msg_to_numpy

class ZedCameraSubscriber:
    def __init__(self, topic_depth='/zedm/zed_node/depth/depth_registered', topic_rgb='/zedm/zed_node/rgb/image_rect_color'):
        self.depth_image = None
        self.rgb_image = None
        self.depth_subscriber = rospy.Subscriber(topic_depth, Image, self.depth_callback, queue_size=2)
        self.rgb_subscriber = rospy.Subscriber(topic_rgb, Image, self.rgb_callback, queue_size=2)

    def depth_callback(self, msg):
        try:
            self.depth_image = image_msg_to_numpy(msg)
        except Exception as e:
            rospy.logerr(f"Depth callback error: {e}")

    def rgb_callback(self, msg):
        try:
            self.rgb_image = image_msg_to_numpy(msg)
        except Exception as e:
            rospy.logerr(f"RGB callback error: {e}")

def main():
    rospy.init_node('zed_pointcloud_visualization')

    # Initialize ZED camera subscriber
    zed_cam = ZedCameraSubscriber()

    # Wait for the first set of images to be received
    rospy.loginfo("Waiting for images from ZED camera...")
    while zed_cam.rgb_image is None or zed_cam.depth_image is None:
        rospy.sleep(0.1)

    # Retrieve camera intrinsic parameters
    camera_info = rospy.wait_for_message('/zedm/zed_node/depth/camera_info', CameraInfo)
    width = camera_info.width
    height = camera_info.height
    fx = camera_info.K[0]
    fy = camera_info.K[4]
    cx = camera_info.K[2]
    cy = camera_info.K[5]

    # Create Open3D intrinsic camera object
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    # Convert ROS images to Open3D format
    color_image = o3d.geometry.Image(zed_cam.rgb_image)
    depth_image = o3d.geometry.Image(zed_cam.depth_image.astype(np.float32))

    # Create RGBD image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image, depth_image, convert_rgb_to_intensity=False)

    # Generate point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

    # Flip the point cloud for correct orientation
    ext_mat = [[0.000737, -0.178996, 0.983850, 0.086292],
               [-0.999998, 0.001475, 0.001017, 0.022357],
               [-0.001633, -0.983849, -0.178995, 0.001017],
               [0., 0., 0., 1.]]
    
    pcd.transform(ext_mat)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    main()
