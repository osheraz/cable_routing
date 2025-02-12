import cv2
import numpy as np
from autolab_core import RigidTransform
from cable_routing.env.ext_camera.zed_camera_pyzed import Zed
import tyro


# Load the ArUco dictionary and parameters
aruco_dict = cv2.aruco.Dictionary_get(
    cv2.aruco.DICT_6X6_100
)  # Change dictionary based on your tags
print(aruco_dict)
aruco_params = cv2.aruco.DetectorParameters_create()


# Known ArUco-to-World transforms
# Define a dictionary mapping ArUco IDs to known ArUco-to-world transforms
# Each entry contains a 4x4 homogeneous transform matrix (R|T)
# [R|T] is a 3x3 rotation matrix (R) and a 3x1 translation vector (T)

TAG_SIZE = 0.098

aruco_to_world = {
    0: np.array(
        [
            [
                0,
                1,
                0,
                0.4360 + 9 * 0.025 - 0.0242 - TAG_SIZE / 2,
            ],  # Transform for ArUco ID 0
            [-1, 0, 0, 0.2600 + 0.059 + TAG_SIZE / 2],
            [0, 0, 1, 0.001293],
            [0, 0, 0, 1],
        ]
    ),
    1: np.array(
        [
            [
                0,
                1,
                0,
                0.4360 + 7 * 0.025 - 0.0242 - TAG_SIZE / 2,
            ],  # Transform for ArUco ID 1
            [-1, 0, 0, 0.2600 - 9 * 0.025 + 0.059 + TAG_SIZE / 2],
            [0, 0, 1, 0.001293],
            [0, 0, 0, 1],
        ]
    ),
    2: np.array(
        [
            [
                0,
                1,
                0,
                0.4360 + 9 * 0.025 - 0.0242 - TAG_SIZE / 2,
            ],  # Transform for ArUco ID 2
            [-1, 0, 0, 0.2600 - 18 * 0.025 + 0.059 + TAG_SIZE / 2],
            [0, 0, 1, 0.001293],
            [0, 0, 0, 1],
        ]
    ),
    3: np.array(
        [
            [
                0,
                1,
                0,
                0.4360 + 9 * 0.025 - 0.0242 - TAG_SIZE / 2,
            ],  # Transform for ArUco ID 3
            [-1, 0, 0, 0.2600 - 27 * 0.025 + 0.059 + TAG_SIZE / 2],
            [0, 0, 1, 0.001293],
            [0, 0, 0, 1],
        ]
    ),
}


def main(is_zed: bool = True, image_path: str = None, tag_size: float = TAG_SIZE):
    # tag size is side length in meters
    # Camera intrinsics (you can obtain these from camera calibration)

    if not is_zed:
        X = np.load("brio_info/config/camera_calibration.npz")  # Example camera matrix
        camera_matrix = X["camera_matrix"]
        dist_coeffs = X["dist_coeffs"]  # Assuming no lens distortion
    else:
        zed = Zed(flip_mode=False, cam_id=None)
        camera_matrix = zed.get_K()
        intr_dict = zed.get_ns_intrinsics()
        dist_coeffs = [intr_dict[key] for key in ["k1", "k2", "p1", "p2"]]

        # dist_coeffs = np.load(
        #     "path_to/zed_dist_coeffs.npy"
        # )  # [intr_dict[key] for key in ["k1", "k2", "p1", "p2"]]

    # Load image
    if image_path is None:
        img_l, _ = zed.get_rgb()
        img = img_l
    else:
        img = cv2.imread(image_path)
    cam = "zed" if is_zed else "brio"
    cv2.imwrite(f"path_to/{cam}_extr.png", img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers in the image
    corners, ids, rejected = cv2.aruco.detectMarkers(
        gray, aruco_dict, parameters=aruco_params
    )
    print(f"{ids} ArUco markers detected")
    if ids is None:
        print("No ArUco markers detected")
        return None

    # Estimate the camera's pose using the detected ArUco tags
    world_points = []  # Points in the world coordinate system
    image_points = []  # Corresponding points in the image
    cv2.aruco.drawDetectedMarkers(img, corners, ids)

    # Estimate pose of each marker and draw 3D axis
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, tag_size, camera_matrix, dist_coeffs
    )
    for i, marker_id in enumerate(ids.flatten()):
        if marker_id in aruco_to_world:

            cv2.drawFrameAxes(
                img, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], tag_size * 0.5
            )

            # Get the 2D image coordinates of the marker corners
            img_points = corners[i].reshape(-1, 2)  # 2D points from the image
            image_points.extend(img_points)

            # Get the corresponding 3D world points from known ArUco-to-world transform
            aruco_world_transform = aruco_to_world[marker_id]

            # Define the 3D points corresponding to the corners of the ArUco tag in the world frame
            tag_corners_world = np.array(
                [
                    [-tag_size / 2, tag_size / 2, 0],
                    [tag_size / 2, tag_size / 2, 0],
                    [tag_size / 2, -tag_size / 2, 0],
                    [-tag_size / 2, -tag_size / 2, 0],
                ]
            )

            # Transform the tag corners to world coordinates using the known ArUco-to-world transform
            for corner in tag_corners_world:
                world_corner_homogeneous = np.append(
                    corner, 1
                )  # Convert to homogeneous coordinates
                world_corner_transformed = (
                    aruco_world_transform @ world_corner_homogeneous
                )
                world_points.append(world_corner_transformed[:3])

    cv2.imshow("Aruco Markers with Axes", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if len(world_points) == 0:
        print("No known ArUco markers detected in the image")
        return None

    world_points = np.array(world_points, dtype=np.float32)
    image_points = np.array(image_points, dtype=np.float32)

    # Estimate the camera extrinsics (rotation and translation)
    success, rvec, tvec = cv2.solvePnP(
        world_points, image_points, camera_matrix, dist_coeffs
    )

    if success:
        # Convert the rotation vector to a rotation matrix
        rot_matrix, _ = cv2.Rodrigues(rvec)

        # Create the extrinsic matrix [R|T]
        extrinsic_matrix = np.hstack((rot_matrix, tvec))  # world to cam
        # concat [0, 0, 0, 1] to last row
        M = np.concatenate([extrinsic_matrix, np.array([[0, 0, 0, 1]])])
        print("cam2world", np.linalg.inv(M))
        cam2world = RigidTransform(
            *RigidTransform.rotation_and_translation_from_matrix(np.linalg.inv(M)),
            from_frame="camera",
            to_frame="world",
        )
        cam2world.save(f"path_to_save/{cam}2world.tf")

        print("Rotation Vector:\n", rvec)
        print("Translation Vector:\n", tvec)
        print("Camera Extrinsic Matrix [R|T]:\n", extrinsic_matrix)

        return extrinsic_matrix
    else:
        print("Pose estimation failed")
        return None


# # Test the function on an image
# image_path = "data/calibrate_extr/frame_0000.jpg"  # Set the path to your image
# calibrate_camera_extrinsics(image_path)

if __name__ == "__main__":
    tyro.cli(main)
