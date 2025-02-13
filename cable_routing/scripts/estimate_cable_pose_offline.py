import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA
import h5py
import cv2
from pathlib import Path
from autolab_core import RigidTransform
from cable_routing.configs.envconfig import BrioConfig
from cable_routing.env.ext_camera.utils.pcl_utils import (
    depth_to_pointcloud,
    project_points_to_image,
    overlay_skeleton_on_image,
)
from cable_routing.env.ext_camera.utils.img_utils import (
    rescale_intrinsics,
    mask_image_outside_roi,
    green_color_segment,
)


def construct_skeletal_graph(cable_points, num_nodes=1000):

    pca = PCA(n_components=1)
    pca.fit(cable_points)
    projected = pca.transform(cable_points).flatten()
    sorted_indices = np.argsort(projected)
    cable_points_sorted = cable_points[sorted_indices]
    node_indices = np.linspace(0, len(cable_points_sorted) - 1, num_nodes, dtype=int)
    cable_nodes = cable_points_sorted[node_indices]
    return cable_nodes


def main():
    script_path = Path(__file__).resolve()

    project_root = script_path.parent.parent.parent
    hdf5_file_path = (
        project_root / "records" / "new" / "camera_data_20250209_161554_0.h5"
    )

    brio_to_world_path = project_root / "data" / "brio" / "brio2world.tf"
    brio_to_world = RigidTransform.load(brio_to_world_path)

    brio_config = BrioConfig()
    brio_intrinsics = brio_config.get_intrinsic_matrix()

    with h5py.File(hdf5_file_path, "r") as hdf:
        brio_rgb = hdf["brio/rgb"][0]

    SCALE_FACTOR = 0.5

    brio_rgb = cv2.resize(
        brio_rgb, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR, interpolation=cv2.INTER_AREA
    )

    # roi = cv2.selectROI(
    #     "Select Region",
    #     cv2.cvtColor(brio_rgb, cv2.COLOR_RGB2BGR),
    #     fromCenter=False,
    #     showCrosshair=True,
    # )
    roi = (0, 0, 0, 0)
    # cv2.destroyWindow("Select Region")
    if roi == (0, 0, 0, 0):
        roi = (0, 0, brio_rgb.shape[1], brio_rgb.shape[0])
    brio_rgb = mask_image_outside_roi(brio_rgb, roi)

    brio_depth = np.full(brio_rgb.shape[:2], 1.0, dtype=np.float32)

    brio_intrinsics = rescale_intrinsics(brio_intrinsics, SCALE_FACTOR)

    points, colors = depth_to_pointcloud(
        brio_depth, brio_rgb, brio_intrinsics, brio_to_world
    )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    cable_points = green_color_segment(pcd)
    cable_nodes = construct_skeletal_graph(cable_points)
    # completed_cable_nodes = apply_cpd(cable_nodes, cable_points)

    cable_pcd = o3d.geometry.PointCloud()
    cable_pcd.points = o3d.utility.Vector3dVector(cable_nodes)
    cable_pcd.paint_uniform_color([0, 1, 0])

    o3d.visualization.draw_geometries([pcd, cable_pcd])

    cable_nodes_2d = project_points_to_image(
        cable_nodes, brio_intrinsics, brio_to_world, brio_rgb.shape[:2]
    )

    result_image = overlay_skeleton_on_image(brio_rgb, cable_nodes_2d)

    cv2.imshow("Projected Skeleton", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
