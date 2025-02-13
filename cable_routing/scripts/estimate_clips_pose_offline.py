import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA
import h5py
from pathlib import Path
from autolab_core import RigidTransform
from cable_routing.env.ext_camera.utils.pcl_utils import depth_to_pointcloud
from cable_routing.configs.envconfig import BrioConfig

import cv2


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


def color_segment_fixtures(point_cloud, color_threshold=(0.1, 0.1, 0.1), display=True):
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


def cluster_fixtures(fixture_points, eps=0.03, min_points=10, display=True):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(fixture_points)

    labels = np.array(
        pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True)
    )

    if labels.size == 0 or (labels == -1).all():
        print("No valid clusters found.")
        return []

    clusters = [fixture_points[labels == i] for i in range(labels.max() + 1) if i != -1]

    if display:
        cluster_pcds = []
        colors = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
        ]

        for i, cluster in enumerate(clusters):
            cluster_pcd = o3d.geometry.PointCloud()
            cluster_pcd.points = o3d.utility.Vector3dVector(cluster)
            cluster_pcd.paint_uniform_color(colors[i % len(colors)])
            cluster_pcds.append(cluster_pcd)

        o3d.visualization.draw_geometries(cluster_pcds)

    return clusters


def classify_fixture(cluster):
    pca = PCA(n_components=3)
    pca.fit(cluster)
    directions = pca.components_
    variances = pca.explained_variance_
    size = np.linalg.norm(cluster.max(axis=0) - cluster.min(axis=0))
    oblongness = variances[0] / variances[1] if variances[1] != 0 else 1.0
    if size < 0.03:
        return "socket-small", directions[0]
    elif size < 0.06:
        return "socket-big", directions[0]
    elif oblongness > 2:
        return "clipper", directions[0]
    else:
        return "retainer", directions[0]


def visualize_results(pcd, clusters, fixture_poses):
    fixture_colors = {
        "socket-small": [1, 0, 0],  # Red
        "socket-big": [0, 1, 0],  # Green
        "retainer": [0, 0, 1],  # Blue
        "clipper": [1, 1, 0],  # Yellow
    }
    cluster_pcds = []
    fixture_orientations = []
    for cluster, (fixture_type, orientation) in zip(clusters, fixture_poses):
        cluster_pcd = o3d.geometry.PointCloud()
        cluster_pcd.points = o3d.utility.Vector3dVector(cluster)
        cluster_pcd.paint_uniform_color(fixture_colors.get(fixture_type, [1, 1, 1]))
        cluster_pcds.append(cluster_pcd)
        centroid = np.mean(cluster, axis=0)
        arrow = o3d.geometry.LineSet()
        arrow.points = o3d.utility.Vector3dVector(
            [centroid, centroid + orientation * 0.05]
        )
        arrow.lines = o3d.utility.Vector2iVector([[0, 1]])
        arrow.colors = o3d.utility.Vector3dVector([[1, 1, 1]])  # White arrow
        fixture_orientations.append(arrow)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    o3d.visualization.draw_geometries([pcd, axis] + cluster_pcds + fixture_orientations)


def main():
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent
    hdf5_file_path = (
        project_root / "records" / "new" / "camera_data_20250209_162231_0.h5"
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

    fixture_points = color_segment_fixtures(pcd)
    clusters = cluster_fixtures(fixture_points)

    fixture_poses = [classify_fixture(cluster) for cluster in clusters]

    visualize_results(pcd, clusters, fixture_poses)


if __name__ == "__main__":
    main()
