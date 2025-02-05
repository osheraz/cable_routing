import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import KDTree

def segment_fixtures(point_cloud, color_threshold=(0.8, 0.2, 0.2)):
    """
    Segments fixtures from the point cloud based on color filtering and clustering.
    """
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)

    # Filter points based on color threshold
    mask = np.linalg.norm(colors - color_threshold, axis=1) < 0.2
    fixture_points = points[mask]

    return fixture_points

def cluster_fixtures(fixture_points, eps=0.02, min_points=10):
    """
    Clusters fixture points using DBSCAN.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(fixture_points)
    
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    clusters = [fixture_points[labels == i] for i in range(labels.max() + 1) if i != -1]
    
    return clusters

def estimate_pose_pca(cluster):
    """
    Estimates the fixture orientation using PCA.
    """
    pca = PCA(n_components=3)
    pca.fit(cluster)
    
    # Principal components
    directions = pca.components_
    shape = pca.explained_variance_ratio_

    # Determine if fixture is oblong or round
    oblongness = shape[0] / shape[1] if shape[1] != 0 else 1.0
    fixture_type = "oblong" if oblongness > 2 else "round"
    
    return fixture_type, directions[0]  # Return the primary axis as orientation

def resolve_ambiguity(start_fixture, cable_points):
    """
    Uses the cable heading direction to resolve fixture orientation ambiguity.
    """
    fixture_position = np.mean(start_fixture, axis=0)
    tree = KDTree(cable_points)
    
    _, nearest_idx = tree.query(fixture_position)
    
    if nearest_idx > 1:
        heading_vector = cable_points[nearest_idx] - cable_points[nearest_idx - 1]
        heading_vector /= np.linalg.norm(heading_vector)
        return heading_vector
    return None

def main():
    # TODO: Load point cloud
    pcd = 

    # Segment fixtures
    fixture_points = segment_fixtures(pcd)

    # Cluster fixtures
    clusters = cluster_fixtures(fixture_points)

    # Analyze each fixture
    fixture_poses = []
    for cluster in clusters:
        fixture_type, orientation = estimate_pose_pca(cluster)
        fixture_poses.append((fixture_type, orientation))

    # cable_pcd = 
    # cable_points = np.asarray(cable_pcd.points)

    # # Resolve ambiguity for "start" and "connector" fixtures
    # for i, (fixture_type, orientation) in enumerate(fixture_poses):
    #     if fixture_type in ["start", "connector"]:
    #         direction = resolve_ambiguity(clusters[i], cable_points)
    #         if direction is not None:
    #             fixture_poses[i] = (fixture_type, direction)

    # Output results
    for i, (fixture_type, orientation) in enumerate(fixture_poses):
        print(f"Fixture {i}: Type={fixture_type}, Orientation={orientation}")

if __name__ == "__main__":
    main()
