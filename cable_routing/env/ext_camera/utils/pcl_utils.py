import open3d as o3d
import numpy as np
import cv2
from autolab_core import RigidTransform


def get_rotation_matrix(roll, pitch, yaw):

    roll, pitch, yaw = np.radians([roll, pitch, yaw])

    R_x = np.array(
        [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]],
        dtype=np.float32,
    )

    R_y = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ],
        dtype=np.float32,
    )

    R_z = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]],
        dtype=np.float32,
    )

    return R_z @ R_y @ R_x


def generate_color_gradient(num_points):
    colors = np.zeros((num_points, 3))
    for i in range(num_points):
        t = i / (num_points - 1) if num_points > 1 else 0
        colors[i] = [1 - t, 0, t]  # Red (start) → Blue (end)
    return colors


def path_to_3d(path, depth_image, intrinsic, ext_mat):
    points_3d = []

    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    for x, y in path:
        z = depth_image[y, x]

        point = np.array([(x - cx) * z / fx, (y - cy) * z / fy, z - 0.05])

        points_3d.append(point)

    points_3d = np.array(points_3d)

    if len(points_3d) > 0:
        points_homo = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
        points_3d = (ext_mat @ points_homo.T).T[:, :3]

    return points_3d


def create_pcd_from_rgbd(rgb_image, depth_image, intrinsic_matrix, ext_mat):
    h, w = depth_image.shape
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")

    x_coords = x_coords.flatten()
    y_coords = y_coords.flatten()
    z_coords = depth_image.flatten()

    valid_mask = z_coords > 0
    x_coords, y_coords, z_coords = (
        x_coords[valid_mask],
        y_coords[valid_mask],
        z_coords[valid_mask],
    )

    x3d = (x_coords - cx) * z_coords / fx
    y3d = (y_coords - cy) * z_coords / fy
    points_3d = np.vstack((x3d, y3d, z_coords)).T

    points_homo = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    points_3d = (ext_mat @ points_homo.T).T[:, :3]

    colors = rgb_image.reshape(-1, 3)[valid_mask] / 255.0

    valid_mask = np.isfinite(points_3d).all(axis=1)

    filtered_points = points_3d[valid_mask]
    filtered_colors = colors[valid_mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

    return pcd


def project_points_to_image(points_3d, intrinsics, extrinsics, image_shape):
    """
    Projects 3D points onto a 2D image.

    Args:
        points_3d (numpy.ndarray): Nx3 array of 3D points.
        intrinsics (numpy.ndarray): 3x3 camera intrinsic matrix.
        extrinsics (RigidTransform): Camera extrinsics (world to camera).
        image_shape (tuple): (height, width) of the image.

    Returns:
        numpy.ndarray: Nx2 array of 2D projected points.
    """
    # Convert to homogeneous coordinates (Nx4)
    points_homo = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))

    # Transform points to camera coordinates
    camera_points = extrinsics.inverse().matrix @ points_homo.T
    camera_points = camera_points[:3, :]

    # Project to 2D using intrinsics
    pixels = intrinsics @ camera_points
    pixels[:2, :] /= pixels[2, :]  # Normalize by depth

    # Convert to pixel coordinates
    pixel_coords = pixels[:2, :].T.astype(int)

    # Filter valid pixels within image bounds
    valid_mask = (
        (pixel_coords[:, 0] >= 0)
        & (pixel_coords[:, 0] < image_shape[1])
        & (pixel_coords[:, 1] >= 0)
        & (pixel_coords[:, 1] < image_shape[0])
    )

    return pixel_coords[valid_mask]


def overlay_skeleton_on_image(image, pixel_coords, color=(0, 255, 0)):
    """
    Draws the projected skeletal graph on the image.

    Args:
        image (numpy.ndarray): Input image.
        pixel_coords (numpy.ndarray): Nx2 array of 2D points.
        color (tuple): RGB color for drawing.

    Returns:
        numpy.ndarray: Image with overlaid skeletal graph.
    """
    output_image = image.copy()

    for point in pixel_coords:
        cv2.circle(output_image, tuple(point), radius=2, color=color, thickness=-1)

    return output_image


def visualize_pointcloud(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    grid = create_grid(grid_size=1.0, grid_spacing=0.1)

    o3d.visualization.draw_geometries([pcd, axis, grid])


def depth_to_pointcloud(depth_map, rgb_image, intrinsics, transform, max_depth=1.2):

    height, width = depth_map.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(width), np.arange(height))
    x = (u - cx) / fx
    y = (v - cy) / fy
    z = depth_map.astype(np.float32)

    valid = z < max_depth
    valid_flat = valid.flatten()

    x, y, z = x[valid] * z[valid], y[valid] * z[valid], z[valid]
    points = np.stack((x, y, z), axis=-1)

    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    points_world = (transform.matrix @ points_homogeneous.T).T[:, :3]

    # rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    rgb_image = rgb_image.astype(np.float32) / 255.0
    rgb_image = rgb_image.reshape(-1, 3)
    colors = rgb_image[valid_flat]

    return points_world, colors


def create_grid(grid_size=1.0, grid_spacing=0.1):
    lines = []
    points = []

    num_lines = int(grid_size / grid_spacing)

    for i in range(-num_lines, num_lines + 1):
        x = i * grid_spacing
        lines.append([len(points), len(points) + 1])
        points.append([x, -grid_size, 0])
        points.append([x, grid_size, 0])

        lines.append([len(points), len(points) + 1])
        points.append([-grid_size, x, 0])
        points.append([grid_size, x, 0])

    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(points)
    grid.lines = o3d.utility.Vector2iVector(lines)

    return grid
