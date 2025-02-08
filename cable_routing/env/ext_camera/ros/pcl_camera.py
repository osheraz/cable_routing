import rospy
import cv2
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sensor_msgs.msg import Image
from image_utils import image_msg_to_numpy
from cable_routing.env.ext_camera.ros.utils.pcl_utils import PointCloudPublisher, PointCloudGenerator


class ZedPointCloudSubscriber:

    def __init__(self,):
        
        # TODO from cfg
        self.far_clip = 0.5
        self.near_clip = 0.1
        self.dis_noise = 0.00
        self.w = 640  
        self.h = 360  

        self.init_success = False
        self.last_cloud = None

        self.pcl_gen = PointCloudGenerator(input_type='depth')

        self.pointcloud_pub = PointCloudPublisher(topic='/results/pointcloud')
        self._rgb_subscriber = rospy.Subscriber('/zedm/zed_node/rgb/image_rect_color', Image, self._rgb_subscriber_callback, queue_size=2)
        self._depth_subscriber = rospy.Subscriber('/zedm/zed_node/depth/depth_registered', Image, self._depth_subscriber_callback, queue_size=2)
        self._check_depth_camera_ready()

        self.timer = rospy.Timer(rospy.Duration(0.01), self.timer_callback)

        self._check_rgb_ready
        self._check_convert_ready()

    def _check_rgb_ready(self):

        self.raw_frame = None
        rospy.logdebug(
            "Waiting for '{}' to be READY...".format('/zedm/zed_node/rgb/image_rect_color'))
        while self.raw_frame is None and not rospy.is_shutdown():
            try:
                self.raw_frame = rospy.wait_for_message(
                    '{}'.format('/zedm/zed_node/rgb/image_rect_color'), Image, timeout=5.0)
                rospy.logdebug(
                    "Current '{}' READY=>".format('/zedm/zed_node/rgb/image_rect_color'))
                self.start_time = rospy.get_time()
                self.raw_frame = image_msg_to_numpy(self.raw_frame)

            except:
                rospy.logerr(
                    "Current '{}' not ready yet, retrying for getting image".format('/zedm/zed_node/rgb/image_rect_color'))
                
        return self.raw_frame
    
    def _check_depth_camera_ready(self):

        self.depth = None
        rospy.logdebug(
            "Waiting for '{}' to be READY...".format('/zedm/zed_node/depth/depth_registered'))
        while self.depth is None and not rospy.is_shutdown():
            try:
                self.depth = rospy.wait_for_message(
                    '{}'.format('/zedm/zed_node/depth/depth_registered'), Image, timeout=5.0)
                rospy.logdebug(
                    "Current '{}' READY=>".format('/zedm/zed_node/depth/depth_registered'))
                self.zed_init = True
                self.depth = image_msg_to_numpy(self.depth)
            except:
                rospy.logerr(
                    "Current '{}' not ready yet, retrying for getting image".format(
                        '/zedm/zed_node/depth/depth_registered'))
        return self.depth
    
    def get_com(self, pcl):

        nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(pcl)
        distances, indices = nbrs.kneighbors(pcl)
        density = np.mean(distances, axis=1)
        weights = 1.0 / density
        weighted_center = np.average(pcl, axis=0, weights=weights)

        return weighted_center

    def _rgb_subscriber_callback(self, msg):
        try:
            self.raw_frame = image_msg_to_numpy(msg)
        except Exception as e:
            print(e)
            return
            
    def _depth_subscriber_callback(self, msg):
        try:
            frame = image_msg_to_numpy(msg)
            self.depth = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_AREA)
        except Exception as e:
            print(e)

    def to_pcl(self, frame):

        try:

            cloud_points = self.pcl_gen.convert(frame)
            proc_cloud = self.process_pointcloud(cloud_points)
            proc_cloud = self.sample_n(proc_cloud, num_sample=4000)
            self.pointcloud_pub.publish_pointcloud(proc_cloud)
            self.last_cloud = proc_cloud

        except Exception as e:
            print(e, 'pcl')

        return self.last_cloud

    def _check_convert_ready(self):
        print('Waiting for depth-to-pcl to init')
        while self.last_cloud is None and not rospy.is_shutdown():
            self.init_success &= self.last_cloud is None
            rospy.sleep(0.1)
        print('depth-to-pcl is ready')

    def sample_n(self, pts, num_sample):
        num = pts.shape[0]
        if num_sample <= num:
            ids = np.random.randint(0, num, size=(num_sample,))
            pts = pts[ids]
        else:
            sampled_pts = pts.copy()
            additional_ids = np.random.randint(0, num, size=(num_sample - num,))
            pts = np.concatenate([sampled_pts, pts[additional_ids]], axis=0)
        return pts

    def process_pointcloud(self, points):

        # x = points[:, 0]
        # y = points[:, 1]
        # z = points[:, 2]
        # valid1 = (z >= -0.1) & (z <= 0.2)
        # valid2 = (x >= 0.2) & (x <= 0.6)
        # valid3 = (y >= -0.4) & (y <= 0.4)

        # valid = valid1 & valid3 & valid2
        # points = points[valid]
        # points = torch.from_numpy(points)
        # sampled_points, indices = ops.sample_farthest_points(points=points.unsqueeze(0), K=points.shape[0])
        # sampled_points = sampled_points.squeeze(0)
        # points = sampled_points.numpy()
        points = self.voxel_grid_sampling(points)

        return points

    def voxel_grid_sampling(self, points, voxel_size=0.001):
        voxel_size_x = voxel_size
        voxel_size_y = voxel_size
        voxel_size_z = voxel_size

        voxel_grid = np.floor(points / np.array([voxel_size_x, voxel_size_y, voxel_size_z])).astype(int)
        unique_voxels, indices = np.unique(voxel_grid, axis=0, return_index=True)
        sampled_points = points[indices]

        return sampled_points

    def get_pcl(self):
        return self.last_cloud

    def get_last_depth(self):

        return self.depth

    def get_last_rgb(self):

        return self.raw_frame
    
    def timer_callback(self, event):
        # Called periodically by the timer
        self.to_pcl(self.depth)


if __name__ == "__main__":
    rospy.init_node('ZedPointCloudPub')
    pcl = ZedPointCloudSubscriber()
    # pointcloud_pub = PointCloudPublisher()
    rate = rospy.Rate(30)

    while not rospy.is_shutdown():
        # pcl.to_object_pcl()
        rate.sleep()