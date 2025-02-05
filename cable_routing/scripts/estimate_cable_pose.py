import open3d as o3d
import numpy as np
from scipy.spatial import KDTree
from sklearn.decomposition import PCA
from pycpd import RigidRegistration

def segment_cable(point_cloud, color_range=((0.9, 0.9, 0.9), (1.0, 1.0, 1.0))):
    """
    Segments the white cable from the point cloud based on color filtering.
    """
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)

    mask = np.all((colors >= color_range[0]) & (colors <= color_range[1]), axis=1)
    cable_points = points[mask]

    return cable_points

def construct_reeb_graph(cable_points, num_nodes=50):
    """
    Constructs a Reeb Graph from the segmented cable points.
    Reduces the cable point cloud into a simplified set of nodes.
    """
    pca = PCA(n_components=1)
    pca.fit(cable_points)
    projected = pca.transform(cable_points).flatten()
    
    sorted_indices = np.argsort(projected)
    cable_points_sorted = cable_points[sorted_indices]

    node_indices = np.linspace(0, len(cable_points_sorted) - 1, num_nodes, dtype=int)
    cable_nodes = cable_points_sorted[node_indices]

    return cable_nodes

def apply_cpd(source_nodes, target_nodes):
    """
    Applies Coherent Point Drift (CPD) to refine the estimated cable nodes.
    """
    reg = RigidRegistration(X=source_nodes, Y=target_nodes)
    transformed_nodes, _ = reg.register()
    return transformed_nodes

def determine_fixture_order(cable_nodes, fixture_positions):
    """
    Determines the order of fixtures based on their closest points along the cable.
    """
    tree = KDTree(cable_nodes)
    fixture_indices = [tree.query(fixture)[1] for fixture in fixture_positions]
    
    sorted_indices = np.argsort(fixture_indices)
    ordered_fixtures = [fixture_positions[i] for i in sorted_indices]

    return ordered_fixtures

def resolve_fixture_orientations(fixtures, cable_nodes):
    """
    Uses the cable heading to resolve fixture orientation ambiguities.
    """
    tree = KDTree(cable_nodes)
    fixture_orientations = []

    for fixture in fixtures:
        _, nearest_idx = tree.query(fixture)
        if nearest_idx > 1:
            heading_vector = cable_nodes[nearest_idx] - cable_nodes[nearest_idx - 1]
            heading_vector /= np.linalg.norm(heading_vector)
            fixture_orientations.append(heading_vector)
        else:
            fixture_orientations.append(None)

    return fixture_orientations

def main():
    
    # TODO
    goal_pcd = 
    peg_board_pcd = 

    cable_points = segment_cable(goal_pcd)

    cable_nodes = construct_reeb_graph(cable_points)

    completed_cable_nodes = apply_cpd(cable_nodes, cable_points)

    fixture_positions = np.array([
        [0.1, 0.2, 0.3],  # TODO
        [0.5, 0.6, 0.7],
        [0.9, 1.0, 1.1]
    ])

    ordered_fixtures = determine_fixture_order(completed_cable_nodes, fixture_positions)

    fixture_orientations = resolve_fixture_orientations(ordered_fixtures, completed_cable_nodes)

    for i, (fixture, orientation) in enumerate(zip(ordered_fixtures, fixture_orientations)):
        print(f"Fixture {i}: Position={fixture}, Orientation={orientation}")

    # Visualize results
    cable_pcd = o3d.geometry.PointCloud()
    cable_pcd.points = o3d.utility.Vector3dVector(completed_cable_nodes)
    cable_pcd.paint_uniform_color([0, 1, 0])  # Green for completed cable

    fixture_pcd = o3d.geometry.PointCloud()
    fixture_pcd.points = o3d.utility.Vector3dVector(np.array(ordered_fixtures))
    fixture_pcd.paint_uniform_color([1, 0, 0])  # Red for fixtures

    o3d.visualization.draw_geometries([cable_pcd, fixture_pcd])

if __name__ == "__main__":
    main()
