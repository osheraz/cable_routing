import h5py
import cv2
import numpy as np
import open3d as o3d
from autolab_core import RigidTransform
from pathlib import Path
from cable_routing.env.ext_camera.utils.pcl_utils import (
    depth_to_pointcloud,
    visualize_pointcloud,
)

near_clip = 0.1
far_clip = 0.4
TABLE_HEIGHT = 0.32


def play_videos_and_project_depth(
    hdf5_file_path,
    brio_dataset_path,
    zed_dataset_path,
    zed_depth_path,
    intrinsics,
    transform,
    fps=10,
):
    try:
        with h5py.File(hdf5_file_path, "r") as hdf:
            if (
                brio_dataset_path not in hdf
                or zed_dataset_path not in hdf
                or zed_depth_path not in hdf
            ):
                print(f"One or more datasets not found in the HDF5 file.")
                return

            brio_dataset = hdf[brio_dataset_path]
            zed_dataset = hdf[zed_dataset_path]
            zed_depth_dataset = hdf[zed_depth_path]

            num_frames = min(
                brio_dataset.shape[0], zed_dataset.shape[0], zed_depth_dataset.shape[0]
            )
            print(
                f"Playing {num_frames} frames and projecting depth to world coordinates..."
            )

            for i in range(2):
                brio_frame = brio_dataset[i]
                zed_frame = zed_dataset[i]
                depth_map = zed_depth_dataset[i]

                depth_image = np.clip(depth_map, near_clip, far_clip)
                depth_image[depth_image >= far_clip] = 0

                points, colors = depth_to_pointcloud(
                    depth_map, zed_frame, intrinsics, transform
                )

                cv2.imshow("zed", zed_frame)
                cv2.imshow("zed depth", (depth_map * 255).astype(np.uint8))
                visualize_pointcloud(points, colors)

                key = cv2.waitKey(int(1000 / fps))
                if key & 0xFF == ord("q"):
                    break

            cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred while processing: {e}")


if __name__ == "__main__":
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent.parent
    hdf5_file_path = (
        project_root / "records" / "new" / "camera_data_20250209_155027_0.h5"
    )

    brio_dataset_path = "brio/rgb"
    zed_dataset_path = "zed/rgb"
    zed_depth_path = "zed/depth"

    zed_intrinsics = np.array(
        [
            [366.24786376953125, 0.0, 323.66802978515625],
            [0.0, 366.24786376953125, 174.6563262939453],
            [0.0, 0.0, 1.0],
        ]
    )

    zed_to_world_path = project_root / "data" / "zed" / "zed2world.tf"
    camera_to_world = RigidTransform.load(zed_to_world_path)

    play_videos_and_project_depth(
        hdf5_file_path,
        brio_dataset_path,
        zed_dataset_path,
        zed_depth_path,
        zed_intrinsics,
        camera_to_world,
    )
