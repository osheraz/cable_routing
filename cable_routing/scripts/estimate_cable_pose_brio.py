import open3d as o3d
import numpy as np
from scipy.spatial import KDTree
from sklearn.decomposition import PCA
from pycpd import RigidRegistration
import h5py
import cv2
from pathlib import Path
from autolab_core import RigidTransform
from cable_routing.env.ext_camera.utils.pcl_utils import depth_to_pointcloud


def rescale_intrinsics(K, scale_factor):
    K_rescaled = K.copy()
    K_rescaled[0, 0] *= scale_factor
    K_rescaled[1, 1] *= scale_factor
    K_rescaled[0, 2] *= scale_factor
    K_rescaled[1, 2] *= scale_factor
    return K_rescaled


def mask_image_outside_roi(image, roi):
    x, y, w, h = roi
    masked_image = image.copy()
    masked_image[:y, :] = 255
    masked_image[y + h :, :] = 255
    masked_image[:, :x] = 255
    masked_image[:, x + w :] = 255
    return masked_image


def color_segment_cable(point_cloud, color_threshold=(0.1, 0.1, 0.1), display=True):
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)
    mask = np.linalg.norm(colors - color_threshold, axis=1) < 0.2
    fixture_points = points[mask]
    if display:
        segmented_pcd = o3d.geometry.PointCloud()
        segmented_pcd.points = o3d.utility.Vector3dVector(fixture_points)
        segmented_pcd.colors = o3d.utility.Vector3dVector(colors[mask])
        o3d.visualization.draw_geometries([segmented_pcd])
    return fixture_points


def construct_reeb_graph(cable_points, num_nodes=50):
    pca = PCA(n_components=1)
    pca.fit(cable_points)
    projected = pca.transform(cable_points).flatten()
    sorted_indices = np.argsort(projected)
    cable_points_sorted = cable_points[sorted_indices]
    node_indices = np.linspace(0, len(cable_points_sorted) - 1, num_nodes, dtype=int)
    cable_nodes = cable_points_sorted[node_indices]
    return cable_nodes


def apply_cpd(source_nodes, target_nodes):
    reg = RigidRegistration(X=source_nodes, Y=target_nodes)
    transformed_nodes, _ = reg.register()
    return transformed_nodes


def main():
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent
    hdf5_file_path = (
        project_root / "records" / "new" / "camera_data_20250209_161554_0.h5"
    )
    brio_to_world_path = project_root / "data" / "brio" / "brio2world.tf"
    brio_to_world = RigidTransform.load(brio_to_world_path)

    brio_intrinsics = np.array(
        [
            [3.43246678e03, 0.0, 1.79637288e03],
            [0.0, 3.44478930e03, 1.08661527e03],
            [0.0, 0.0, 1.0],
        ]
    )

    with h5py.File(hdf5_file_path, "r") as hdf:
        brio_rgb = hdf["brio/rgb"][0]

    SCALE_FACTOR = 0.5
    brio_rgb = cv2.resize(
        brio_rgb, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR, interpolation=cv2.INTER_AREA
    )

    roi = cv2.selectROI("Select Region", brio_rgb, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Region")
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

    cable_points = color_segment_cable(pcd)
    cable_nodes = construct_reeb_graph(cable_points)
    # completed_cable_nodes = apply_cpd(cable_nodes, cable_points)

    cable_pcd = o3d.geometry.PointCloud()
    cable_pcd.points = o3d.utility.Vector3dVector(cable_nodes)
    cable_pcd.paint_uniform_color([0, 1, 0])

    o3d.visualization.draw_geometries([pcd, cable_pcd])


if __name__ == "__main__":
    main()
