import h5py
import cv2
import numpy as np

def play_videos_side_by_side(hdf5_file_path, brio_dataset_path, zed_dataset_path, fps=10):
    """
    Play BRIO and ZED camera videos side by side from an HDF5 file.

    Parameters:
    - hdf5_file_path: Path to the HDF5 file.
    - brio_dataset_path: Path within the HDF5 file to the BRIO dataset.
    - zed_dataset_path: Path within the HDF5 file to the ZED dataset.
    - fps: Frames per second for video playback.
    """
    try:
        with h5py.File(hdf5_file_path, 'r') as hdf:
            if brio_dataset_path not in hdf or zed_dataset_path not in hdf:
                print(f"One or both datasets '{brio_dataset_path}' and '{zed_dataset_path}' not found in the HDF5 file.")
                return

            brio_dataset = hdf[brio_dataset_path]
            zed_dataset = hdf[zed_dataset_path]

            num_frames = min(brio_dataset.shape[0], zed_dataset.shape[0])
            print(f"Playing {num_frames} frames from each dataset...")

            for i in range(num_frames):
                brio_frame = brio_dataset[i]
                zed_frame = zed_dataset[i]

                if brio_frame.ndim == 3 and brio_frame.shape[2] == 3:
                    brio_frame = cv2.cvtColor(brio_frame, cv2.COLOR_RGB2BGR)
                if zed_frame.ndim == 3 and zed_frame.shape[2] == 3:
                    zed_frame = cv2.cvtColor(zed_frame, cv2.COLOR_RGB2BGR)

                # Resize frames to the same height
                height = min(brio_frame.shape[0], zed_frame.shape[0])
                brio_frame = cv2.resize(brio_frame, (int(brio_frame.shape[1] * height / brio_frame.shape[0]), height))
                zed_frame = cv2.resize(zed_frame, (int(zed_frame.shape[1] * height / zed_frame.shape[0]), height))

                # Concatenate frames side by side
                combined_frame = np.hstack((brio_frame, zed_frame))

                cv2.imshow('BRIO and ZED Video Playback', combined_frame)
                if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                    break
            
            print(num_frames)
            cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred while playing the videos: {e}")

if __name__ == "__main__":
    # Replace with the path to your HDF5 file
    hdf5_file_path = '/home/osheraz/cable_routing/records/camera_data_20250206_174759_0.h5'
    # Paths to the datasets within the HDF5 file
    brio_dataset_path = 'brio/rgb'
    zed_dataset_path = 'zed/rgb'

    play_videos_side_by_side(hdf5_file_path, brio_dataset_path, zed_dataset_path)
