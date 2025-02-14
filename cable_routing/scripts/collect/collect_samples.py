import os
import rospy
import h5py
import numpy as np
from datetime import datetime
from cable_routing.env.ext_camera.ros.brio_subscriber import BRIOSubscriber
from cable_routing.env.ext_camera.ros.zed_camera import ZedCameraSubscriber


class CameraDataCollector:
    def __init__(
        self,
        save_directory,
        file_prefix="camera_data",
        max_file_size=100,
        with_brio=False,
        with_zed=True,
    ):
        rospy.init_node("camera_data_collector", anonymous=True)
        self.save_directory = save_directory
        self.file_prefix = file_prefix
        self.max_file_size = max_file_size * 1024 * 1024
        self.with_brio = with_brio
        self.with_zed = with_zed

        self.brio_camera = BRIOSubscriber() if with_brio else None
        self.zed_camera = ZedCameraSubscriber() if with_zed else None

        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)

        self.hdf5_file = None
        self.current_file_size = 0
        self.file_index = 0
        self._create_new_hdf5_file()

    def _create_new_hdf5_file(self):
        if self.hdf5_file:
            self.hdf5_file.close()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.file_prefix}_{timestamp}_{self.file_index}.h5"
        filepath = os.path.join(self.save_directory, filename)
        self.hdf5_file = h5py.File(filepath, "w")
        self.current_file_size = 0
        self.file_index += 1

        if self.with_brio:
            bwidth_, height_ = 3809, 2121
            brio_group = self.hdf5_file.create_group("brio")
            self.brio_rgb_dataset = brio_group.create_dataset(
                "rgb",
                shape=(0, height_, bwidth_, 3),
                maxshape=(None, height_, bwidth_, 3),
                chunks=(1, height_, bwidth_, 3),
                dtype=np.uint8,
                compression="gzip",
                compression_opts=9,
            )

        if self.with_zed:
            zed_group = self.hdf5_file.create_group("zed")
            self.zed_rgb_dataset = zed_group.create_dataset(
                "rgb",
                shape=(0, self.zed_camera.h, self.zed_camera.w, 3),
                maxshape=(None, self.zed_camera.h, self.zed_camera.w, 3),
                chunks=(1, self.zed_camera.h, self.zed_camera.w, 3),
                dtype=np.uint8,
                compression="gzip",
                compression_opts=9,
            )

            self.zed_depth_dataset = zed_group.create_dataset(
                "depth",
                shape=(0, self.zed_camera.h, self.zed_camera.w),
                maxshape=(None, self.zed_camera.h, self.zed_camera.w),
                chunks=(1, self.zed_camera.h, self.zed_camera.w),
                dtype=np.float32,
                compression="gzip",
                compression_opts=9,
            )

    def _append_to_dataset(self, dataset, data):
        dataset.resize((dataset.shape[0] + 1), axis=0)
        dataset[-1] = data
        self.current_file_size += data.nbytes

    def collect_and_save_data(self):
        rate = rospy.Rate(30)

        while not rospy.is_shutdown():
            brio_rgb = self.brio_camera.get_frame() if self.with_brio else None
            zed_rgb, zed_depth = (
                self.zed_camera.get_frames() if self.with_zed else (None, None)
            )

            if self.with_brio and brio_rgb is not None:
                self._append_to_dataset(self.brio_rgb_dataset, brio_rgb)

            if self.with_zed:
                if zed_rgb is not None:
                    self._append_to_dataset(self.zed_rgb_dataset, zed_rgb)
                if zed_depth is not None:
                    self._append_to_dataset(self.zed_depth_dataset, zed_depth)

            if self.current_file_size >= self.max_file_size:
                self._create_new_hdf5_file()

            rate.sleep()

    def shutdown(self):
        if self.hdf5_file:
            self.hdf5_file.close()
            print("File saved successfully.")


if __name__ == "__main__":
    save_dir = "/home/osheraz/cable_routing/records"
    collector = CameraDataCollector(
        save_directory=save_dir, max_file_size=100, with_brio=True, with_zed=True
    )
    try:
        collector.collect_and_save_data()
    except rospy.ROSInterruptException:
        pass
    finally:
        collector.shutdown()
