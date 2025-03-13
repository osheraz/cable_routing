import os
import rospy
import h5py
import numpy as np
import json
from std_msgs.msg import String
from datetime import datetime
from cable_routing.env.ext_camera.ros.brio_subscriber import BRIOSubscriber
from cable_routing.env.ext_camera.ros.zed_camera import ZedCameraSubscriber


class CameraDataCollector:
    def __init__(
        self,
        save_directory,
        file_prefix="camera_data",
        max_file_size=1000,
        with_brio=True,
        with_zed=True,
        zed_img_count=None,
        board=None,
        plan=None,
    ):
        rospy.init_node("camera_data_collector", anonymous=True)
        self.save_directory = save_directory
        self.file_prefix = file_prefix
        self.max_file_size = max_file_size * 1024 * 1024
        self.with_brio = with_brio
        self.with_zed = with_zed
        self.zed_img_count = zed_img_count
        self.board = board
        self.plan = plan or []

        self.recording = False  # Flag to control recording
        self.brio_camera = BRIOSubscriber() if with_brio else None
        self.zed_camera = ZedCameraSubscriber() if with_zed else None

        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)

        self.hdf5_file = None
        self.current_file_size = 0
        self.file_index = 0

        rospy.Subscriber("/camera_recorder/command", String, self.command_callback)

    def command_callback(self, msg):
        if msg.data == "start":
            if not self.recording:
                rospy.loginfo("Starting recording...")
                self.recording = True
                self._create_new_hdf5_file()
        elif msg.data == "stop":
            if self.recording:
                rospy.loginfo("Stopping recording...")
                self.recording = False
                self.shutdown()

    def _create_new_hdf5_file(self):
        if self.hdf5_file:
            self.hdf5_file.close()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.file_prefix}_{timestamp}_{self.file_index}.h5"
        filepath = os.path.join(self.save_directory, filename)
        self.hdf5_file = h5py.File(filepath, "w")
        self.current_file_size = 0
        self.file_index += 1

        if self.board:
            board_json = json.dumps(self.board.get_clips())
            self.hdf5_file.create_dataset("board", data=board_json)

        if self.plan:
            plan_json = json.dumps(self.plan)
            self.hdf5_file.create_dataset("plan", data=plan_json)

        if self.with_brio:
            bwidth_, height_ = 1920, 1080
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

    def _append_to_dataset(self, dataset, data):
        dataset.resize((dataset.shape[0] + 1), axis=0)
        dataset[-1] = data
        self.current_file_size += data.nbytes

    def collect_and_save_data(self):
        rate = rospy.Rate(30)
        zed_count = 0

        while not rospy.is_shutdown():
            if self.recording:
                brio_rgb = self.brio_camera.get_frame() if self.with_brio else None
                zed_rgb = self.zed_camera.get_rgb() if self.with_zed else None

                if self.with_brio and brio_rgb is not None:
                    self._append_to_dataset(self.brio_rgb_dataset, brio_rgb)

                if (
                    self.with_zed
                    and zed_rgb is not None
                    and zed_count <= self.zed_img_count
                ):
                    self._append_to_dataset(self.zed_rgb_dataset, zed_rgb)
                    zed_count += 1

                if self.current_file_size >= self.max_file_size:
                    self._create_new_hdf5_file()

            rate.sleep()

    def shutdown(self):
        if self.hdf5_file:
            self.hdf5_file.close()
            rospy.loginfo("File saved successfully.")


if __name__ == "__main__":
    save_dir = "/home/osheraz/cable_routing/records"

    class DummyBoard:
        def get_clips(self):
            return {"clip_1": [1, 2, 3], "clip_2": [4, 5, 6]}

    board = DummyBoard()
    plan = ["Step 1", "Step 2", "Step 3"]

    collector = CameraDataCollector(
        save_directory=save_dir,
        max_file_size=1000,
        with_brio=True,
        with_zed=True,
        zed_img_count=1,
        board=board,
        plan=plan,
    )

    collector.collect_and_save_data()

    # rostopic pub /camera_recorder/command std_msgs/String "start"
    # rostopic pub /camera_recorder/command std_msgs/String "stop"
