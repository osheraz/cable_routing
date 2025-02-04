import cv2
import numpy as np
import glob
from autolab_core import CameraIntrinsics

def calibrate_camera(image_dir, chessboard_size, output_calib, output_intr):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    
    objpoints, imgpoints = [], []
    images = glob.glob(image_dir)
    
    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if found:
            objpoints.append(objp)
            imgpoints.append(cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria))
    
    if not objpoints or not imgpoints:
        raise ValueError("No valid chessboard corners found.")
    
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    np.savez(output_calib, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, rvecs=rvecs, tvecs=tvecs)
    
    intr = CameraIntrinsics(
        frame="camera",
        fx=camera_matrix[0, 0], fy=camera_matrix[1, 1],
        cx=camera_matrix[0, 2], cy=camera_matrix[1, 2],
        skew=0, height=gray.shape[0], width=gray.shape[1]
    )
    intr.save(output_intr)
    
    print("Calibration completed and saved.")

def main():
    image_dir = "data/calibrate_intr/*.jpg"
    chessboard_size = (8, 6)
    output_calib = "path/camera_calibration.npz"
    output_intr = "path/camera_intr.intr"
    calibrate_camera(image_dir, chessboard_size, output_calib, output_intr)

if __name__ == "__main__":
    main()
