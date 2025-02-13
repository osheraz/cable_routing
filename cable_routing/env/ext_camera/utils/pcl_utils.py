import open3d as o3d
import numpy as np
import cv2


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
