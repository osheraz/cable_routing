# Copyright 2024 Nimrod Curtis
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
import os
import subprocess
import json
import sys

class BagCompress:
    """Class for compressing ROS bag files.

    This class provides static methods to compress ROS bag files either in a single batch folder or
    across multiple batch folders within a given directory.
    """

    def __init__(self) -> None:
        """Initializes the BagCompress class."""
        pass
    
    @staticmethod
    def compress_batch(bag_folder_path: str):
        """Compresses all ROS bag files in a specified folder.

        Args:
            bag_folder_path (str): Path to the folder containing ROS bag files to compress.
        """
        print(f"[INFO] compressing batch - {bag_folder_path}")
        
        
        metadata_file_p = os.path.join(bag_folder_path,"metadata.json")

        try:
            if not os.path.exists(metadata_file_p):
                print(f"'metadata.json' not found in {bag_folder_path}.")
                sys.exit("Exiting program.")
        except FileNotFoundError as e:
            print(e)

        # If the file exists, read and return its contents
        with open(metadata_file_p, 'r') as file:
            metadata = json.load(file)

        # Loop through each file in the folder
        for filename in os.listdir(bag_folder_path):
            if (filename.endswith(".bag")):
                if not(filename in metadata["compressed_bags"]):
                    # Compress the file using rosbag compress command
                    subprocess.call(["rosbag", "compress", os.path.join(bag_folder_path, filename)])
                    original_file_name = filename.rsplit('.')[0] + ".orig.bag"
                    os.remove(os.path.join(bag_folder_path, original_file_name))
                
                    # Add to metadata
                    metadata["compressed_bags"].append(filename)
                else:
                    print(f"[INFO] {filename} already compressed")
        
        # Writing the updated data back to the file
        with open(metadata_file_p, 'w') as file:
            json.dump(metadata, file, indent=4)

    @staticmethod
    def compress_folder(folder_path: str):
        """Compresses ROS bag files in all batch folders within a specified directory.

        Args:
            folder_path (str): Path to the directory containing batch folders of ROS bag files.
        """
        print(f"[INFO] compressing folder - {folder_path}")
        
        # Loop through each folder
        for filename in os.listdir(folder_path):
            if filename.startswith("bag_batch"):
                batch_path = os.path.join(folder_path, filename)
                BagCompress.compress_batch(batch_path)


    