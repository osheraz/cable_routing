import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA
import h5py
from pathlib import Path
from autolab_core import RigidTransform
from cable_routing.env.ext_camera.utils.pcl_utils import depth_to_pointcloud
import cv2


def color_segment_fixtures(point_cloud, color_threshold=(0.1, 0.1, 0.1)):
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)

    mask = np.linalg.norm(colors - color_threshold, axis=1) < 0.2
    fixture_points = points[mask]

    return fixture_points


def cluster_fixtures(fixture_points, eps=0.02, min_points=10):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(fixture_points)

    labels = np.array(
        pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True)
    )

    if labels.size == 0 or (labels == -1).all():  # Check if all points are noise
        print("No valid clusters found.")
        return []

    clusters = [fixture_points[labels == i] for i in range(labels.max() + 1) if i != -1]

    return clusters


def estimate_pose_pca(cluster):
    pca = PCA(n_components=3)
    pca.fit(cluster)
    directions = pca.components_
    shape = pca.explained_variance_ratio_
    oblongness = shape[0] / shape[1] if shape[1] != 0 else 1.0
    fixture_type = "oblong" if oblongness > 2 else "round"
    return fixture_type, directions[0]


def visualize_results(pcd, clusters, fixture_poses):

    fixture_colors = [
        [1, 0, 0],  # Red
        [0, 1, 0],  # Green
        [0, 0, 1],  # Blue
        [1, 1, 0],  # Yellow
        [1, 0, 1],  # Magenta
        [0, 1, 1],  # Cyan
    ]

    cluster_pcds = []
    fixture_orientations = []

    for i, cluster in enumerate(clusters):
        cluster_pcd = o3d.geometry.PointCloud()
        cluster_pcd.points = o3d.utility.Vector3dVector(cluster)
        cluster_pcd.paint_uniform_color(fixture_colors[i % len(fixture_colors)])
        cluster_pcds.append(cluster_pcd)

        fixture_type, orientation = fixture_poses[i]
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
        project_root / "records" / "new" / "camera_data_20250209_155027_0.h5"
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

    SCALE_FACTOR = 0.25
    brio_rgb = cv2.resize(
        brio_rgb, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR, interpolation=cv2.INTER_AREA
    )

    brio_depth = np.full(brio_rgb.shape[:2], 1.0, dtype=np.float32)

    def rescale_intrinsics(K, scale_factor):
        K_rescaled = K.copy()
        K_rescaled[0, 0] *= scale_factor  # Scale fx
        K_rescaled[1, 1] *= scale_factor  # Scale fy
        K_rescaled[0, 2] *= scale_factor  # Scale cx
        K_rescaled[1, 2] *= scale_factor  # Scale cy
        return K_rescaled

    brio_intrinsics = rescale_intrinsics(brio_intrinsics, SCALE_FACTOR)
    points, colors = depth_to_pointcloud(
        brio_depth, brio_rgb, brio_intrinsics, brio_to_world
    )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    fixture_points = color_segment_fixtures(pcd)
    clusters = cluster_fixtures(fixture_points)

    fixture_poses = []
    for cluster in clusters:
        fixture_type, orientation = estimate_pose_pca(cluster)
        fixture_poses.append((fixture_type, orientation))

    visualize_results(pcd, clusters, fixture_poses)


if __name__ == "__main__":
    main()
