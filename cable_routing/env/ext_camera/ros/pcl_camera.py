import rospy
from sensor_msgs.msg import PointCloud2, CameraInfo, Image
import matplotlib.pyplot as plt
from image_utils import image_msg_to_numpy
from pcl_utils import PointCloudPublisher

import torch
import cv2
from sklearn.neighbors import NearestNeighbors
import numpy as np
from zed_camera import ZedCameraSubscriber

np.set_printoptions(suppress=True, formatter={'float_kind': '{: .3f}'.format})


class PointCloudGenerator:
    def __init__(self, 
                 camera_info_topic='/zedm/zed_node/depth/camera_info',
                 sample_num=None,  depth_max=None, input_type='depth', device='cpu'):


        if camera_info_topic is not None:
            camera_info = rospy.wait_for_message(camera_info_topic, CameraInfo)

            # Extract camera properties from CameraInfo message
            self.cam_width = camera_info.width
            self.cam_height = camera_info.height
            self.fu = camera_info.K[0]  # fx
            self.fv = camera_info.K[4]  # fy
            self.cu = camera_info.K[2]  # cx
            self.cv = camera_info.K[5]  # cy

            # Extract projection matrix (3x4)
            proj_matrix_ros = camera_info.P
            self.proj_matrix = torch.tensor(proj_matrix_ros).reshape(3, 4).to(device)

        else:
            # Fallback if no ROS info is available
            raise ValueError('Camera info not available')

        # * -1 fu
        self.int_mat = torch.Tensor(
            [[self.fu, 0, self.cu],
             [0, self.fv, self.cv],
             [0, 0, 1]]
        )
        

        # TODO get from calib
        self.ext_mat = torch.tensor([[0.000737, -0.178996, 0.983850, 0.086292],
                                     [-0.999998, 0.001475, 0.001017, 0.022357],
                                 [-0.001633, -0.983849, -0.178995, 0.001017],
                                     [0., 0., 0., 1.]], dtype=torch.float32).to(device)

        def get_rotation_matrix(roll, pitch, yaw):
            roll, pitch, yaw = torch.tensor(roll), torch.tensor(pitch), torch.tensor(yaw)
            R_x = torch.tensor(
                [[1, 0, 0], [0, torch.cos(roll), -torch.sin(roll)], [0, torch.sin(roll), torch.cos(roll)]],
                dtype=torch.float32)
            R_y = torch.tensor(
                [[torch.cos(pitch), 0, torch.sin(pitch)], [0, 1, 0], [-torch.sin(pitch), 0, torch.cos(pitch)]],
                dtype=torch.float32)
            R_z = torch.tensor([[torch.cos(yaw), -torch.sin(yaw), 0], [torch.sin(yaw), torch.cos(yaw), 0], [0, 0, 1]],
                               dtype=torch.float32)
            return torch.mm(R_z, torch.mm(R_y, R_x))

        # roll, pitch, yaw = 0.00, 0.0, 0.0
        # rotation_tilt = get_rotation_matrix(roll, pitch, yaw)
        # new_rot = torch.mm(rotation_tilt, self.ext_mat[:3, :3])
        # self.ext_mat = torch.cat([torch.cat([new_rot, self.ext_mat[:3, 3:]], dim=1), self.ext_mat[3:, :]], dim=0).T

        self.int_mat_T_inv = torch.inverse(self.int_mat.T).to(device)
        self.depth_max = depth_max

        x, y = torch.meshgrid(torch.arange(self.cam_height), torch.arange(self.cam_width))
        self._uv_one = torch.stack((y, x, torch.ones_like(x)), dim=-1).float().to(device)

        self._uv_one_in_cam = self._uv_one @ self.int_mat_T_inv

        self._uv_one_in_cam = self._uv_one_in_cam.repeat(1, 1, 1)
        self.sample_num = sample_num
        self.device = device

        self.input_type = input_type

    def convert(self, points):
        # Process depth buffer
        points = torch.tensor(points, device=self.device, dtype=torch.float32)

        if self.input_type == 'depth':
            if self.depth_max is not None:
                valid_ids = points > self.depth_max
            else:
                valid_ids = torch.ones(points.shape, dtype=bool, device=self.device)

            valid_depth = points[valid_ids]  # TODO
            uv_one_in_cam = self._uv_one_in_cam[valid_ids]

            # Calculate 3D points in camera coordinates
            pts_in_cam = torch.mul(uv_one_in_cam, valid_depth.unsqueeze(-1))
        else:
            # already pcl
            pts_in_cam = points

        # R_flip_y_up = torch.tensor([
        #     [0, -1, 0],
        #     [1, 0, 0],
        #     [0, 0, 1]
        # ], dtype=torch.float32, device=points.device)
        #
        # # Apply the rotation to convert to Y-up coordinate system
        # pts_in_cam = torch.matmul(pts_in_cam[:, :3], R_flip_y_up.T)

        # plot_point_cloud(pts_in_cam.cpu().detach().numpy())

        pts_in_cam = torch.cat((pts_in_cam,
                                torch.ones(*pts_in_cam.shape[:-1], 1,
                                           device=pts_in_cam.device)),
                               dim=-1)

        pts_in_world = torch.matmul(pts_in_cam, self.ext_mat)

        pcd_pts = pts_in_world[:, :3]
        # plot_point_cloud(pcd_pts.cpu().detach().numpy())

        return pcd_pts.cpu().detach().numpy()


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