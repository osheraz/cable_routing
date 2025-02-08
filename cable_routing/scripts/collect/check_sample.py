import h5py
import cv2
import numpy as np
import open3d as o3d
from autolab_core import RigidTransform
from pathlib import Path

near_clip = 0.1
far_clip = 0.4
TABLE_HEIGHT = 0.32

def process_depth_image(depth_image):
    depth_image = np.clip(depth_image, near_clip, far_clip)
    depth_image[depth_image >= far_clip] = 0 
    return depth_image

def depth_to_pointcloud(depth_map, intrinsics, transform):
    height, width = depth_map.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(width), np.arange(height))
    x = (u - cx) / fx
    y = (v - cy) / fy
    z = depth_map.astype(np.float32) 

    valid = z > TABLE_HEIGHT
    x, y, z = x[valid] * z[valid], y[valid] * z[valid], z[valid]
    points = np.stack((x, y, z), axis=-1)

    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    points_world = (transform.matrix @ points_homogeneous.T).T[:, :3]

    colors = np.zeros_like(points)
    colors[:, 2] = (z - z.min()) / (z.max() - z.min())

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


def visualize_pointcloud(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    grid = create_grid(grid_size=1.0, grid_spacing=0.1)

    o3d.visualization.draw_geometries([pcd, axis, grid])


def play_videos_and_project_depth(hdf5_file_path, brio_dataset_path, zed_dataset_path, zed_depth_path, intrinsics, transform, fps=10):
    try:
        with h5py.File(hdf5_file_path, 'r') as hdf:
            if brio_dataset_path not in hdf or zed_dataset_path not in hdf or zed_depth_path not in hdf:
                print(f"One or more datasets not found in the HDF5 file.")
                return

            brio_dataset = hdf[brio_dataset_path]
            zed_dataset = hdf[zed_dataset_path]
            zed_depth_dataset = hdf[zed_depth_path]

            num_frames = min(brio_dataset.shape[0], zed_dataset.shape[0], zed_depth_dataset.shape[0])
            print(f"Playing {num_frames} frames and projecting depth to world coordinates...")

            for i in range(2):
                brio_frame = brio_dataset[i]
                zed_frame = zed_dataset[i]
                depth_map = zed_depth_dataset[i]
                depth_map = process_depth_image(depth_map)

                points, colors = depth_to_pointcloud(depth_map, intrinsics, transform)

                cv2.imshow('zed', zed_frame)
                cv2.imshow('zed depth', (depth_map * 255).astype(np.uint8))
                visualize_pointcloud(points, colors)

                key = cv2.waitKey(int(1000 / fps))
                if key & 0xFF == ord('q'):
                    break

            cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred while processing: {e}")


if __name__ == "__main__":
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent.parent
    hdf5_file_path = project_root / "records" / "camera_data_20250206_174445_0.h5"

    brio_dataset_path = 'brio/rgb'
    zed_dataset_path = 'zed/rgb'
    zed_depth_path = 'zed/depth'

    zed_intrinsics = np.array([
        [700.0, 0, 320], 
        [0, 700.0, 180],
        [0, 0, 1]         
    ])

    zed_to_world_path = project_root / "data" / "zed" / "zed2world.tf"
    camera_to_world = RigidTransform.load(zed_to_world_path)

    play_videos_and_project_depth(hdf5_file_path, brio_dataset_path, zed_dataset_path, zed_depth_path, zed_intrinsics, camera_to_world)
