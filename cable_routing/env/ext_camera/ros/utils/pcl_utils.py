import rospy
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sensor_msgs.msg import Header, PointCloud2, PointField, CameraInfo
import sensor_msgs.point_cloud2 as pc2

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


def remove_statistical_outliers(points, k=20, z_thresh=2.0):
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(points)  # k+1 because the point itself is included
    distances, _ = nbrs.kneighbors(points)
    mean_distances = np.mean(distances[:, 1:], axis=1)
    mean = np.mean(mean_distances)
    std = np.std(mean_distances)
    inliers = np.where(np.abs(mean_distances - mean) < z_thresh * std)[0]

    return points[inliers]


def plot_point_cloud(points):
    """ Visualize 3D point cloud using Matplotlib """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = points[::10, 0]
    y = points[::10, 1]
    z = points[::10, 2]
    ax.scatter(x, y, z, c=z, cmap='viridis', marker='.')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


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


class PointCloudPublisher:
    def __init__(self, topic='pointcloud'):
        self.pcl_pub = rospy.Publisher(f'/{topic}', PointCloud2, queue_size=10)

    def publish_pointcloud(self, points):
        """
        Publish the point cloud to a ROS topic.

        :param points: numpy array of shape [N, 3] representing the point cloud
        """
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'base_link'  # Set the frame according to your setup

        # Define the PointCloud2 fields (x, y, z)
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)
        ]

        # Convert the numpy array to PointCloud2 format
        cloud_msg = pc2.create_cloud(header, fields, points)

        # Publish the point cloud message
        self.pcl_pub.publish(cloud_msg)