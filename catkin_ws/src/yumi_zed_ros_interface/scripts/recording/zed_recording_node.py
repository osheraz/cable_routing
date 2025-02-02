#!/usr/bin/env python3

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard library imports
import os
import yaml
import datetime
import json

# ROS application/library specific imports
import rospy
import rosnode
import rosparam
from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse, EmptyRequest
from sensor_msgs.msg import CameraInfo
from tf2_msgs.msg import TFMessage
from ros_bag_recorder import RosBagRecord
from rospy_message_converter import message_converter
from zed_interfaces.srv import reset_odometry

PATH = os.path.dirname(__file__)

class ZedRecordManager(object):
    """
    Manages the recording of ZED camera data in ROS, including parameters and camera information.

    This class interfaces with a ROS bag recorder to manage the recording of camera data and associated 
    ROS topics. It also handles service requests for starting and stopping the recording.

    Attributes:
        rosbag_recorder (RosBagRecord): Object to manage ROS bag recording.
        ctrl_c (bool): Flag to indicate if the script is exiting.
        _is_recording (bool): Flag to track if the recording is currently active.
        _ros_start_time (rospy.Time): The start time of the recording.
        _ros_current_time (rospy.Time): The current ROS time.
        _metadata (dict): Dictionary which hold a meta information on the folder.
    """

    def __init__(self) -> None:
        """
        Initializes the ZedRecordManager with the given configuration.
        """
        
        # Initialize parameters
        params = self._load_parameters()

        # Store the topics to a list 
        topics_list_keys = params["topics_to_rec"]
        topics = params["topics"]
        topics_list = [topics.get(k) for k in topics_list_keys]

        self.record_script = os.path.join(PATH, params["script"])           # use bash script from path in config
        
        bag_folder = os.path.join(PATH, params["bag_folder"])
        prefix = params["prefix"]
        # Generate the folder name with prefix and current date-time
        current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        folder_name = f"{prefix}_{current_time}"
        
        # Complete path for the record folder
        self.record_folder = os.path.join(bag_folder, folder_name)

        # Create the folder if it doesn't exist
        if not os.path.exists(self.record_folder):
            os.makedirs(self.record_folder, exist_ok=True)       

        self.recorded_configs_folder = os.path.join(self.record_folder, "configs")        
        if not os.path.exists(self.recorded_configs_folder):
            os.makedirs(self.recorded_configs_folder)

        # Initialize a RosBagRecord instance
        self.rosbag_recorder = RosBagRecord(topics_list=topics_list,
                                        record_script_path=self.record_script,
                                        record_folder=self.record_folder,
                                        record_node_name="record")


        # Initialize additional flags and attributes 
        self.ctrl_c = False
        rospy.on_shutdown(self.shutdownhook)

        self._target_obj_topic = topics.get("target_object") if "target_object" in topics_list_keys else None
        self._is_recording = False
        self._ros_start_time = rospy.Time.now()
        self._ros_current_time = rospy.Time.now()
        self._record_counter = 0
        
        self._metadata = {}
        self._metadata["time"] = {}
        self._metadata["time"]["start"] = current_time
        self._metadata["processed_bags"] = []
        self._metadata["compressed_bags"] = []

    def spin(self):
        """
        Main loop of the node. Checks for ZED node status and handles recording.
        """
        
        zed_node = "/zed/zed_node"
        # Use rosnode to get the list of running nodes
        running_nodes = rosnode.get_node_names()
        
        try:
            while (not (zed_node in running_nodes)) and not(self.ctrl_c) :
                rospy.loginfo_throttle(2,"{} is not runnning. Please check camera".format(zed_node))
                running_nodes = rosnode.get_node_names()

            if (not self.ctrl_c and (zed_node in running_nodes)):
                rospy.loginfo("{} is runnning properly.".format(zed_node))
                self._save_zed_params()
                self._save_camera_info()
                self._save_recording_params()
                self.recording_service = rospy.Service("~record",SetBool,self._record_callback)

                while not rospy.is_shutdown():
                    if self._is_recording:
                        self._ros_current_time = rospy.Time.now()
                        dt = (self._ros_current_time - self._ros_start_time).to_sec()
                        rospy.loginfo_throttle(10,"Recording for {:.2f} [secs]".format(dt))

            rospy.spin()
        
        except rospy.ROSException as e:
            rospy.logwarn("ZED node didn't start")
            self.rosbag_recorder.stop_recording_handler(record_running=self._is_recording)

    def shutdownhook(self):
        """
        Function to be called on shutdown. Stops recording and performs cleanup.
        """
        
        self.ctrl_c = True
        self.rosbag_recorder.stop_recording_handler(record_running=self._is_recording)

        # Save metadta 
        self._save_metadata()
        
        # Cleanup actions before exiting
        rospy.logwarn("Shutting down " + rospy.get_name())

    def _load_parameters(self):
        """
        Loads and returns various recording-related parameters from the ROS Parameter Server.

        Returns:
            params (dict): A dictionary containing the loaded parameters.
        """
        
        node_name = rospy.get_name()
        
        params = {}
        params["topics"] = rosparam.get_param(node_name+"/topics")
        params["topics_to_rec"] = rosparam.get_param(node_name+"/recording/topics_to_rec")
        params["bag_folder"] = rosparam.get_param(node_name+"/recording/bag_folder")
        params["script"] = rosparam.get_param(node_name+"/recording/script")
        params["prefix"] = rosparam.get_param(node_name+"/recording/prefix")

        return params
        
        
    def _record_callback(self,request:SetBoolRequest):
        """
        Callback function for the recording service.

        Args:
            request (SetBoolRequest): The service request containing the command to start or stop recording.

        Returns:
            SetBoolResponse: The response indicating success or failure and an accompanying message.
        """
        
        if request.data:
            if not self._is_recording:
                self.rosbag_recorder.start()
                self._record_counter+=1
                self._is_recording = True
                self._ros_start_time = rospy.Time.now()
            else:
                return SetBoolResponse(success=False, message="Already recording.")

        elif not request.data:
            if self._is_recording:
                self.rosbag_recorder.stop_recording_handler(record_running=self._is_recording)
                self._is_recording = False
                self._ros_current_time = rospy.Time.now()
                dt = (self._ros_current_time - self._ros_start_time).to_sec()
                rospy.loginfo("Recording stopped after {:.2f} seconds.".format(dt))
            else:
                return SetBoolResponse(success=False, message="Already not recording.")

        if request.data:
            response_message = "Received recording command, start recording."
        else:
            self._ros_current_time = rospy.Time.now()
            dt = (self._ros_current_time - self._ros_start_time).to_sec()
            response_message = "Received stop command, recording stopped after {:.2f} seconds.".format(dt)

        return SetBoolResponse(success=True, message=response_message)

    def _save_zed_params(self):
        """
        Saves the ZED camera parameters to a file.
        """
        
        rospy.loginfo("Saving camera params")
        if not os.path.exists(self.recorded_configs_folder):
            os.mkdir(self.recorded_configs_folder)
        rosparam.dump_params(self.recorded_configs_folder+"/zed.yaml",param="/zed")

    def _save_camera_info(self):
        """
        Saves camera info and static TFs to a file.
        """
        
        rospy.loginfo("Saving camera info and static TFs")
        if not os.path.exists(self.recorded_configs_folder):
            os.mkdir(self.recorded_configs_folder)

        camera_info_left = rospy.wait_for_message("/zed/zed_node/left/camera_info",CameraInfo,timeout=5)
        camera_info_right = rospy.wait_for_message("/zed/zed_node/right/camera_info",CameraInfo,timeout=5)
        info_dic={}

        tf_static_imu = rospy.wait_for_message("/tf_static",TFMessage,timeout=0.1)
        while len(tf_static_imu.transforms)!=1:
            tf_static_imu = rospy.wait_for_message("/tf_static",TFMessage,timeout=0.1)

        tf_static_base_to_optical = rospy.wait_for_message("/tf_static",TFMessage,timeout=0.1)
        while len(tf_static_base_to_optical.transforms)==1:
            tf_static_base_to_optical = rospy.wait_for_message("/tf_static",TFMessage,timeout=0.1)

        info_dic["tf_static_imu"] = message_converter.convert_ros_message_to_dictionary(tf_static_imu)
        info_dic["tf_static_base_to_optical"] = message_converter.convert_ros_message_to_dictionary(tf_static_base_to_optical)
        info_dic["left_camera_info"] = message_converter.convert_ros_message_to_dictionary(camera_info_left)
        info_dic["right_camera_info"] = message_converter.convert_ros_message_to_dictionary(camera_info_right)
        file_path = os.path.join(self.recorded_configs_folder,"camera_info.yaml")
        with open(file_path, 'w') as file:
            yaml.dump(info_dic, file)
    
    def _save_recording_params(self):
        """
        Saves recording parameters to a file.
        """

        rospy.loginfo("Saving recording parameters")
        if not os.path.exists(self.recorded_configs_folder):
            os.mkdir(self.recorded_configs_folder)
        rosparam.dump_params(self.recorded_configs_folder+"/record.yaml",param=rospy.get_name())
        
        # if target detection is on -> save the detection params
        if self._target_obj_topic is not None:
            rosparam.dump_params(self.recorded_configs_folder+"/obj_detect_converter.yaml",
                                    param="/"+self._target_obj_topic.split('/')[1])
        
    def _save_metadata(self):
        """
        Saves metadata with some relevant info
        """
        rospy.loginfo("Saving metadata")

        current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self._metadata["time"]["end"] = current_time
        self._metadata["bags_number"] = self._record_counter
        
        file_path = os.path.join(self.record_folder,"metadata.json")
        
        # Writing the dictionary to a JSON file
        with open(file_path, 'w') as json_file:
            json.dump(self._metadata, json_file, indent=4)

def main():

    # Main function implementation

    try:
        rospy.init_node("zed_recording_node")                # Init node
        rospy.loginfo("********** Starting node {} **********".format(rospy.get_name()))
        zed_recorder_manager = ZedRecordManager()
        zed_recorder_manager.spin()

    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
