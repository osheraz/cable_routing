import h5py
import cv2
import numpy as np
import open3d as o3d
from autolab_core import RigidTransform
from pathlib import Path
from cable_routing.configs.envconfig import ZedMiniConfig

from cable_routing.env.ext_camera.utils.pcl_utils import (
    depth_to_pointcloud,
    visualize_pointcloud,
)

near_clip = 0.1
far_clip = 0.4
TABLE_HEIGHT = 0.32


def play_videos_and_project_depth(
    hdf5_file_path,
    brio_dataset_path=None,
    zed_dataset_path=None,
    zed_depth_path=None,
    intrinsics=None,
    transform=None,
    fps=10,
):
    try:
        with h5py.File(hdf5_file_path, "r") as hdf:
            if zed_dataset_path and zed_dataset_path not in hdf:
                print(f"ZED dataset not found in the HDF5 file.")
                return

            if zed_depth_path and zed_depth_path not in hdf:
                print(f"ZED depth dataset not found in the HDF5 file.")
                return

            brio_dataset = (
                hdf[brio_dataset_path]
                if brio_dataset_path and brio_dataset_path in hdf
                else None
            )
            zed_dataset = hdf[zed_dataset_path] if zed_dataset_path else None
            zed_depth_dataset = hdf[zed_depth_path] if zed_depth_path else None

            num_frames = min(
                brio_dataset.shape[0] if brio_dataset else float("inf"),
                zed_dataset.shape[0] if zed_dataset else float("inf"),
                zed_depth_dataset.shape[0] if zed_depth_dataset else float("inf"),
            )
            print(
                f"Playing {num_frames} frames and projecting depth to world coordinates..."
            )

            for i in range(num_frames):
                brio_frame = brio_dataset[i] if brio_dataset is not None else None
                zed_frame = zed_dataset[i] if zed_dataset is not None else None
                depth_map = (
                    zed_depth_dataset[i] if zed_depth_dataset is not None else None
                )

                if depth_map is not None:
                    depth_image = np.clip(depth_map, near_clip, far_clip)
                    depth_image[depth_image >= far_clip] = 0

                    points, colors = depth_to_pointcloud(
                        depth_map, zed_frame, intrinsics, transform
                    )
                    visualize_pointcloud(points, colors)

                if zed_frame is not None:
                    cv2.imshow("zed", zed_frame)

                if depth_map is not None:
                    cv2.imshow("zed depth", (depth_map * 255).astype(np.uint8))

                if brio_frame is not None:
                    cv2.imshow("brio", brio_frame)

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

    zed_config = ZedMiniConfig()
    zed_intrinsic = zed_config.get_intrinsic_matrix()

    zed_to_world_path = project_root / "data" / "zed" / "zed_to_world.tf"
    camera_to_world = RigidTransform.load(zed_to_world_path)

    play_videos_and_project_depth(
        hdf5_file_path,
        brio_dataset_path,
        zed_dataset_path,
        zed_depth_path,
        zed_intrinsic,
        camera_to_world,
    )
