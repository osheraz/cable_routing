import os
import rospy
import h5py
import numpy as np
from datetime import datetime
from cable_routing.env.ext_camera.ros.brio_publisher import CameraPublisher
from cable_routing.env.ext_camera.ros.zed_camera import ZedCameraSubscriber

class CameraDataCollector:
    def __init__(self, save_directory, file_prefix='camera_data', max_file_size=100):
        """
        Initialize the data collector.

        Parameters:
        - save_directory: Directory where HDF5 files will be saved.
        - file_prefix: Prefix for the HDF5 filenames.
        - max_file_size: Maximum size of each HDF5 file in MB.
        """
        rospy.init_node('camera_data_collector', anonymous=True)
        self.save_directory = save_directory
        self.file_prefix = file_prefix
        self.max_file_size = max_file_size * 1024 * 1024  # Convert MB to bytes

        self.brio_camera = CameraPublisher()
        self.zed_camera = ZedCameraSubscriber()

        # Create save directory if it doesn't exist
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)

        # Initialize HDF5 file
        self.hdf5_file = None
        self.current_file_size = 0
        self.file_index = 0
        self._create_new_hdf5_file()

    def _create_new_hdf5_file(self):
        """Create a new HDF5 file for data storage."""
        if self.hdf5_file:
            self.hdf5_file.close()

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{self.file_prefix}_{timestamp}_{self.file_index}.h5"
        filepath = os.path.join(self.save_directory, filename)
        self.hdf5_file = h5py.File(filepath, 'w')
        self.current_file_size = 0
        self.file_index += 1

        # Create groups for BRIO and ZED data
        self.brio_group = self.hdf5_file.create_group('brio')
        self.zed_group = self.hdf5_file.create_group('zed')

        # Initialize datasets with maxshape to allow appending
        self.brio_rgb_dataset = self.brio_group.create_dataset(
            'rgb',
            shape=(0, self.brio_camera.height, self.brio_camera.width, 3),
            maxshape=(None, self.brio_camera.height, self.brio_camera.width, 3),
            dtype=np.uint8,
            compression="gzip",
            compression_opts=9
        )

        self.zed_rgb_dataset = self.zed_group.create_dataset(
            'rgb',
            shape=(0, self.zed_camera.h, self.zed_camera.w, 3),
            maxshape=(None, self.zed_camera.h, self.zed_camera.w, 3),
            dtype=np.uint8,
            compression="gzip",
            compression_opts=9
        )

        self.zed_depth_dataset = self.zed_group.create_dataset(
            'depth',
            shape=(0, self.zed_camera.h, self.zed_camera.w),
            maxshape=(None, self.zed_camera.h, self.zed_camera.w),
            dtype=np.float32,
            compression="gzip",
            compression_opts=9
        )

    def _append_to_dataset(self, dataset, data):
        """Append data to an HDF5 dataset."""
        dataset.resize((dataset.shape[0] + 1), axis=0)
        dataset[-1] = data
        self.current_file_size += data.nbytes

    def collect_and_save_data(self):
        """Collect data from cameras and save to HDF5 files."""
        rate = rospy.Rate(10)  # 10 Hz

        while not rospy.is_shutdown():
            brio_rgb = self.brio_camera.get_frame()
            zed_rgb, zed_depth = self.zed_camera.get_frames()

            if brio_rgb is not None:
                self._append_to_dataset(self.brio_rgb_dataset, brio_rgb)
            if zed_rgb is not None:
                self._append_to_dataset(self.zed_rgb_dataset, zed_rgb)
            if zed_depth is not None:
                self._append_to_dataset(self.zed_depth_dataset, zed_depth)

            # Check if current file exceeds the maximum size
            if self.current_file_size >= self.max_file_size:
                self._create_new_hdf5_file()

            rate.sleep()

    def shutdown(self):
        """Close the HDF5 file and release resources."""
        if self.hdf5_file:
            self.hdf5_file.close()
        self.brio_camera.stop()

if __name__ == '__main__':
    save_dir = '/path/to/save/directory'  # Specify your desired save directory
    collector = CameraDataCollector(save_directory=save_dir, max_file_size=100)  # 100 MB per file
    try:
        collector.collect_and_save_data()
    except rospy.ROSInterruptException:
        pass
    finally:
        collector.shutdown()
