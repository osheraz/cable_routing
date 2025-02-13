import open3d as o3d
import numpy as np
import cv2
from autolab_core import RigidTransform


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
