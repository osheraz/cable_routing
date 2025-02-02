#!/usr/bin/env python3


#
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
import subprocess
import os

# ROS application/library specific imports
import rospy

class RosBagRecord:
    """
    A class for managing the recording of ROS topics into a bag file.

    This class provides methods to start and stop the recording of specified ROS topics.
    It also includes functionality to terminate specific ROS nodes.

    Attributes:
        record_folder (str): The directory where the recorded bag file will be stored.
        command (str): The command to be executed for recording the topics.
    """

    def __init__(self, topics_list, record_script_path, record_folder, record_node_name="record"):
        """
        Initializes the RosBagRecord object with parameters for recording.

        Args:
            topics_list (list of str): A list of ROS topics to be recorded.
            record_script_path (str): The file path to a script that initiates the recording process.
            record_folder (str): The directory where the recorded bag file will be stored.
        """
        self.record_folder = record_folder
        self.record_node_name = record_node_name
        self.command = "source " + record_script_path + " " + " ".join(topics_list) + " __name:=" + self.record_node_name # Build the rosbag command based on the topics list

    def start(self):
        """
        Starts the recording of ROS topics into a bag file.

        This method runs the recording command in a separate process within the specified record folder.
        """
        # Start the recording in a subprocess
        self.p = subprocess.Popen(self.command, 
                                  stdin=subprocess.PIPE,
                                  cwd=self.record_folder,
                                  shell=True,
                                  executable='/bin/bash')

    def terminate_ros_node(self, s):
        """
        Terminates ROS nodes that start with a specified string.

        Args:
            s (str): The prefix of the node names to be terminated.
        """
        # List all ROS nodes and terminate the specified ones
        list_cmd = subprocess.Popen("rosnode list", shell=True, stdout=subprocess.PIPE)
        list_output = list_cmd.stdout.read()
        retcode = list_cmd.wait()
        assert retcode == 0, "List command returned %d" % retcode
        for str in list_output.split(b"\n"):
            str_decode = str.decode('utf8')
            if str_decode.startswith(s):
                os.system("rosnode kill " + str_decode)  # Kill the node

    def stop_recording_handler(self, record_running=True):
        """
        Stops the recording process and handles the saving of the recorded data.

        Args:
            record_running (bool): A flag to indicate whether recording is currently running. Defaults to True.
        """
        
        # Terminate recording if it's running
        ns = rospy.get_namespace()
        if record_running:
            self.terminate_ros_node(ns+self.record_node_name)
            rospy.loginfo("Saving bag to " + self.record_folder)
        else:
            rospy.loginfo("Record didn't run. No recording to terminate.")
