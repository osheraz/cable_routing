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
import json
import sys
import os
import yaml

# Third party libraries
import h5py
import matplotlib.pyplot as plt
import moviepy.video.io.VideoFileClip as mp
from tqdm import tqdm
from matplotlib import animation

# ROS libraries
import rosbag
from cv_bridge import CvBridge

# Custom libraries
from bagtool.process.utils import *
class BadReader:
    """
    A class to read and process data from a ROS bag file.

    Attributes:
        bagfile (str): Path to the ROS bag file.
        filename (str): Name of the bag file.
        dir (str): Directory of the bag file.
        dst_datafolder (str): Destination folder for processed data.
        metadata (dict): Metadata information of the bag file.
        topics (list): List of topics in the bag file.
        topics_to_keys (dict): Mapping of topics to their respective keys.
        raw_data (dict): Raw data extracted from the bag file.
        aligned_data (dict): Aligned data based on synchronization rate.
        sync_rate (float): Synchronization rate for data alignment.
        A (4X4 np.array): Transformation matrix from world origin to start pose.
    
    Methods:
        __init__: Initializes the BadReader instance.
        _init_raw_data: Initializes the raw data structure.
        _init_aligned_data: Initializes the aligned data structure.
        _get_sync_rate: Calculates the synchronization rate.
        _process_data: Processes the data from the bag file.
        get_raw_element: Retrieves raw data elements.
        get_aligned_element: Retrieves aligned data elements.
        get_raw_dataset: Returns the raw data dataset.
        get_aligned_dataset: Returns the aligned data dataset.
        save_raw: Saves raw data to a file.
        save_aligned: Saves aligned data to files.
        save_traj_video: Generates and saves a trajectory video.
    """
    def __init__(self, bagfile, dst_dataset = None, dst_datafolder_name=None, config = None, mode = "data") -> None:
        """
        Initializes a BagReader object to process ROS bag files.

        This constructor sets up the necessary attributes for reading and processing a ROS bag file. It extracts metadata, 
        topics information, and initializes data structures for storing raw and aligned data.

        Args:
            bagfile (str): Path to the ROS bag file to be read.
            dst_dataset (str, optional): Destination dataset path where processed data folders will be created. 
                                        Defaults to the directory of the bag file if None.
            dst_datafolder_name (str, optional): Base name of the folder to store processed data. This will be appended 
                                                with a suffix derived from the bag file name. Defaults to 'bag-data' if None.
            config (dict, optional): Configuration dictionary which may specify options like sync_rate, 
                                    aligned_topics, pre_truncated, and post_truncated. Defaults to None.

        """
        
        print(f"\n[INFO]  Reading {bagfile}.")

        self.bagfile = bagfile
        self.mode = mode
        parts = bagfile.split('/')
        # If the bag_file contains '/', parts will have more than one element
        if len(parts) > 1:
            self.filename = parts[-1]
            self.dir = '/'.join(parts[:-1])
        else:
            self.filename = bagfile
            self.dir = './'

        # Set the pathes and names for the bag data folder in the destination data folder
        data_folder_name_suffix = 'bag-'+self.filename[0:-4]+'-data'
        data_folder_name = dst_datafolder_name + '_' + data_folder_name_suffix if dst_datafolder_name is not None \
                                            else data_folder_name_suffix 

        # If no destination mentioned -> open store the data folder in the current directory
        if dst_dataset is not None:
            self.dst_datafolder = os.path.join(dst_dataset,data_folder_name)
        else:
            self.dst_datafolder = os.path.join(self.dir,data_folder_name)

        # Make the directory
        if os.path.exists(self.dst_datafolder):
                print(f"[INFO]  Data folder {self.dst_datafolder} already exists. Not creating.")
        else:
            try:
                os.mkdir(self.dst_datafolder)
            except OSError:
                print(f"[ERROR] Failed to create the data folder {self.dst_datafolder}.")
                sys.exit("Exiting program.")
            else:
                print(f"[INFO]  Successfully created the data folder {self.dst_datafolder}.")
        
        # Initialize a metadata file for the destination data folder
        metadata_file_p = os.path.join(self.dst_datafolder,"metadata.json")
        self.metadata = {}
        self.metadata["source_filename"] = self.filename
        self.metadata["source_dir"] = self.dir
        self.metadata["data_dir"] = self.dst_datafolder

        # Open the record config file
        record_config_file = os.path.join(self.dir,
                                        "configs",
                                        "record.yaml")
        with open(record_config_file, 'r') as file:
            record_config = yaml.safe_load(file)
        self.metadata['demonstrator'] = record_config['recording']['demonstrator']
        
        # Init a Bag reader instance
        try:
            self.reader = rosbag.Bag(self.bagfile)
        except rosbag.ROSBagException as e:
            print(e)
            print(f"Error loading {self.bagfile}.")
            sys.exit("Exiting program.")

        # Start to collect some info from the bag  
        info = self.reader.get_type_and_topic_info()
        topic_tuple = info.topics.values()
        self.topics = info.topics.keys()
        self.topics_to_keys = {}
        
        message_types = []
        for t1 in topic_tuple: message_types.append(t1.msg_type)

        n_messages = []
        for t1 in topic_tuple: n_messages.append(t1.message_count)

        frequency = []
        for t1 in topic_tuple: frequency.append(t1.frequency)
        
        title=['Topic', 'Type', 'Message Count', 'Frequency']
        topics_zipped = list(zip(self.topics,message_types, n_messages, frequency))

        
        


        # Initialize the 'topics' dictionary in metadata
        self.metadata['topics'] = {}

        # Iterate over topics_zipped and populate the metadata
        for topic_data in topics_zipped:
            topic_key = get_key_by_value(record_config["topics"],topic_data[0])

            # Get info per topic
            self.metadata['topics'][topic_key] = dict(zip(title, topic_data))
            self.topics_to_keys[topic_data[0]] = topic_key

        # Initialize the raw & aligned (sync) data containers
        self.raw_data = self._init_raw_data()
        self.aligned_data = self._init_aligned_data(aligned_topics = config.get("aligned_topics") if config is not None \
                                                    else None)

        # Set the sync rate and max_depth
        self.sync_rate = config.get("sync_rate", self._get_sync_rate()) if config is not None else self._get_sync_rate()
        self.max_depth = config.get("max_depth", 10000) if config is not None else 10000 # 10 [meter]
        self.metadata['sync_rate'] = self.sync_rate
        self.metadata['max_depth'] = self.max_depth
        
        # Init the homogenouse matrix from the world to start pose
        self.A = None

        # Process the data
        self._process_data(sync_rate=self.sync_rate,
                        pre_truncated=config.get("pre_truncated"),
                        post_truncated=config.get("post_truncated"))

        # Calculate statistics

        # statistics = self._stats_calc(aligned_data=self.aligned_data, frame_rate=self.sync_rate)
        
        self.metadata['num_of_synced_msgs'] = len(self.aligned_data['dt'])
        self.metadata['time'] = self.aligned_data['time_elapsed'][-1]
        # self.metadata['stats'] = statistics

        # if self.mode == "eval":
        #     results = self._results(data=self.aligned_data)
        #     self.metadata["results"] = results
        
        print(f"[INFO]  Saving metadata.")
        with open(metadata_file_p, 'w') as file:
            json.dump(self.metadata, file, indent=4)

    #TODO: move this out
    def _stats_calc(self, aligned_data, frame_rate):
        
        data = aligned_data
        dpos_arr = np.array([data["topics"]["odom"][i]['relative_frame']['position'] for i in range(len(data["topics"]["odom"]))])
        dyaw_arr = np.array([data["topics"]["odom"][i]['relative_frame']['yaw'] for i in range(len(data["topics"]["odom"]))])

        dpos_max = np.max(dpos_arr, axis=0)*frame_rate
        dpos_min = np.min(dpos_arr, axis=0)*frame_rate
        dpos_mean = np.mean(dpos_arr, axis=0)*frame_rate
        dpos_std = np.std(dpos_arr, axis=0)*frame_rate
        
        dyaw_max = np.max(dyaw_arr)*frame_rate
        dyaw_min = np.min(dyaw_arr)*frame_rate
        dyaw_mean = np.mean(dyaw_arr)*frame_rate
        dyaw_std = np.std(dyaw_arr)*frame_rate

        statistics = {
            'dx': {
                'max': dpos_max[0],
                'min': dpos_min[0],
                'mean': dpos_mean[0],
                'std': dpos_std[0]
            },
            'dy': {
                'max': dpos_max[1],
                'min': dpos_min[1],
                'mean': dpos_mean[1],
                'std': dpos_std[1]
            },
            'dyaw': {
                'max': dyaw_max,
                'min': dyaw_min,
                'mean': dyaw_mean,
                'std': dyaw_std
            }
        }
        
        return statistics

    def _results(self,data):
        
        res = {}
        success = bool(sum(data["topics"]["goal_reach"]) > 0)
        
        res["success"] = success
        
        return res

    def _init_raw_data(self):
        """
        Initialize the structure for storing raw data extracted from the bag file.

        This method sets up a dictionary to store time and data for each recorded topic in the bag file.

        Returns:
            dict: A dictionary with keys for each topic and sub-keys for 'time' and 'data'.
        """
        dic = {}
        topics_recorded_keys = self.metadata['topics'].keys()
        
        for topic in topics_recorded_keys:
            dic[topic] = {}
            dic[topic]['time'] = []
            dic[topic]['data'] = []

        return dic

    def _init_aligned_data(self, aligned_topics=['odom','rgb']):
        """
        Initialize the structure for storing aligned data based on specified topics.

        This method sets up a dictionary to store delta time, time elapsed, and data for each aligned topic.

        Args:
            aligned_topics (list of str): List of topics to be aligned. Defaults to ['odom', 'rgb'].

        Returns:
            dict: A dictionary with keys for delta time, time elapsed, and each aligned topic.
        """
        dic = {}
        dic['dt'] = []
        dic['time_elapsed'] = []
        dic['topics'] = {}
        topics_aligned_keys = aligned_topics

        for tk in topics_aligned_keys:
            dic['topics'][tk] = []

        return dic

    def _get_sync_rate(self):
        """
        Determine the synchronization rate for aligning data from different topics.

        Returns:
            float: The synchronization rate for data alignment.
        """
        min_freq = np.inf
        
        for tk in self.aligned_data['topics'].keys():
            if tk in self.metadata['topics'].keys():
                freq = self.metadata['topics'][tk]['Frequency']
                
                if freq < min_freq:
                    min_freq = freq

        return min_freq


    def _process_data(self, sync_rate, pre_truncated, post_truncated):
        """
        Processes messages from the ROS bag file to extract and align data.

        This method reads messages from the bag file, extracts raw data for specified topics, and aligns them at a given synchronization rate (sync_rate).
        It updates the `raw_data` and `aligned_data` dictionaries with the extracted and aligned data, respectively.

        Args:
            sync_rate (float): The rate (in Hz) at which data should be aligned.
            pre_truncated (float): Time (in seconds) to truncate from the start of the recordings to avoid initial noise.
            post_truncated (float): Time (in seconds) to truncate from the end of the recordings to avoid end artifacts.

        Updates:
            self.raw_data: Dictionary to store raw data.
            self.aligned_data: Dictionary to store data aligned at the specified sync_rate.
        """

        # Initialize start and end times of the bag file
        starttime = self.reader.get_start_time()
        end_time = self.reader.get_end_time()
        currtime = starttime  # current time starts at the beginning
        
        # Iterate through each message in the ROS bag
        for topic, msg, t in self.reader.read_messages(topics=self.topics):
            topic_key = self.topics_to_keys[topic]  # Get the unique topic key

            if topic_key in self.aligned_data["topics"].keys():  # Check if the topic is designated for alignment
                self.raw_data[topic_key]['time'].append(t.to_sec())  # Store raw timestamp
                data = self.get_raw_element(topic=topic_key, msg=msg)
                self.raw_data[topic_key]['data'].append(data)  # Store raw data

            # Check if it's time to align data
            if topic_key in self.aligned_data["topics"].keys() and ((t.to_sec() - currtime) >= 1.0 / sync_rate):
                # Verify if current time is within the specified truncated periods
                if ((currtime > starttime + pre_truncated) and (currtime < end_time - post_truncated) and
                    (t.to_sec() > starttime + pre_truncated) and (t.to_sec() < end_time - post_truncated)):

                    for tk in self.aligned_data['topics'].keys():  # Process each topic for alignment
                        data_entry = self.raw_data[tk]['data'][-1]  # Last raw data entry
                        data_aligned = self.get_aligned_element(topic=tk, data=data_entry)  # Align data
                        self.aligned_data['topics'][tk].append(data_aligned)  # Append aligned data

                currtime = t.to_sec()  # Update current time
                # Compute the elapsed time since the last alignment
                if len(self.aligned_data['time_elapsed']) > 0:
                    prevtime = self.aligned_data['time_elapsed'][-1]
                    self.aligned_data['dt'].append(currtime - prevtime)
                else:
                    self.aligned_data['dt'].append(0)

                self.aligned_data['time_elapsed'].append(currtime - starttime)  # Append total elapsed time

    def get_raw_element(self,topic,msg):
        """
        Retrieve a raw data element based on the topic and message.

        This method processes a ROS message from a given topic and converts it into a usable format.

        Args:
            topic (str): The topic of the data.
            msg (rosbag message): The ROS message to be processed.

        Returns:
            Varies: The processed data element, the type depends on the topic.
        """

        switcher = {
            # "depth": image_compressed_to_numpy,
            "depth": image_to_numpy,
            "rgb": image_to_numpy,
            "odom": odom_to_numpy,
            "target_object": object_detection_to_dic,
            "goal_reach": lambda x: x.data,
            "goal_pose": pose_to_numpy, 
        }

        if topic == "depth":
            case_function = switcher.get(topic, self.max_depth)
        else:
            case_function = switcher.get(topic)

        return case_function(msg)

    def get_aligned_element(self,topic,data):
        """
        Process and align a data element based on the topic.

        Args:
            topic (str): The topic of the data.
            data: The data element to be processed.

        Returns:
            Varies: The aligned data element, the type depends on the topic.
        """
        switcher = {
            "depth": lambda x: x,
            "rgb": lambda x: x,
            "odom": np_odom_to_xy_yaw,
            "target_object": lambda x:x,
            "goal_reach": lambda x:x,
            "goal_pose": np_pose_in_odom
        }

        case_function = switcher.get(topic)
        
        ## TODO: refactore!!!
        if(topic == "odom"):
            prev_data = self.aligned_data['topics'][topic][-1] if len(self.aligned_data['topics'][topic])>0 else None
            
            if prev_data is None:
                x_start, y_start, yaw_start = data[0][0], data[0][1], quat_to_yaw(data[1])
                self.A = get_transform_to_start(x_start, y_start, yaw_start)

            return case_function(data, prev_data, self.A)
        elif(topic == "goal_pose"):
            return case_function(data, self.A)
        else:
            return case_function(data)

    def get_raw_dataset(self):
        """
        Return the entire raw data set.

        Returns:
            dict: The raw data set.
        """
        return self.raw_data

    def get_aligned_dataset(self):
        """
        Return the entire aligned data set.

        This method provides access to the data aligned based on the synchronization rate.

        Returns:
            dict: The aligned data set.
        """
        return self.aligned_data

    def save_raw(self, data:dict):
        """
        Save the raw data to files in a specified format.

        This method writes the raw data to files, organizing them based on the topics.

        Args:
            data (dict): The raw data to be saved.
        """

        folder_path = os.path.join(self.dst_datafolder,'raw_data')
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
            for tk,v in data.items():
                file_path = os.path.join(folder_path,f'{tk}_raw.h5')

                # Create a HDF5 file
                with h5py.File(file_path, 'w') as h5file:
                    # Store the time
                    h5file.create_dataset('time', data=v['time'])
                    
                    # Store the data depend on the topic
                    if tk in ["odom", "goal_pose"]:
                        
                        data_arrays = {}
                        num_keys = len(v['data'][0])  # Assuming all data entries have the same number of elements

                        # Initialize arrays for each key with the correct length
                        # indexes correspond to : ["position","orientation","linear_vel","angular_vel"]
                        for i in range(num_keys):
                            data_arrays[i] = []

                        # Process each entry in v['data'] only once
                        for data_entry in v['data']:
                            for i in range(num_keys):
                                data_arrays[i].append(data_entry[i])

                        # Create datasets for each key
                        for i in range(num_keys):
                            h5file.create_dataset(f'data_{i}', data=np.array(data_arrays[i]))

                    elif tk in ['rgb', 'depth']: 
                        h5file.create_dataset('data', data=np.array(v['data']))

                    elif tk == "target_object":
                        if len(v['data'])>0:
                            data_arrays = {}
                            keys = list(v['data'][0].keys())  # Assuming all data entries have the same number of elements
                            
                            # Initialize arrays for each key with the correct length
                            for key in keys:
                                data_arrays[key] = []

                            # Process each entry in v['data'] only once
                            for data_entry in v['data']:
                                for key in keys:
                                    data_arrays[key].append(data_entry[key])

                            # Create datasets for each key
                            for key in keys:
                                h5file.create_dataset(f'data_{key}', data = data_arrays[key] if type(data_arrays[key][0]==str) else np.array(data_arrays[key]) )

                h5file.close()
                print(f"[INFO]  {tk} raw data successfully saved.")

    def save_aligned(self, data:dict):
        """
        Save the aligned data to files in a specified format.

        This method writes the aligned data to files, organizing them based on the topics.

        Args:
            data (dict): The aligned data to be saved.
        """

        for tk in data['topics'].keys():  # Iterate over each topic in the aligned data
            if tk == 'odom':
                # Filename and path for odometry data
                filename = 'traj_robot_data'
                file_path = os.path.join(self.dst_datafolder, filename + '.json')
                
                # Initialize dictionary to store odometry data
                dic_to_save = {}
                for frame in ['gt_frame', 'odom_frame', 'relative_frame']:
                    dic_to_save.update({frame: {'position': [], 'yaw': []}})
                    for type in ['position', 'yaw']:
                        for i in range(len(data['topics'][tk])):
                            dic_to_save[frame][type].append(data['topics']['odom'][i][frame][type])

                # Write odometry data to a JSON file
                with open(file_path, 'w') as file:
                    json.dump(dic_to_save, file, indent=4)
            
            elif tk == 'goal_pose':
                # Filename and path for odometry data
                filename = 'goal_pose_data'
                file_path = os.path.join(self.dst_datafolder, filename + '.json')
                
                # Initialize dictionary to store odometry data
                dic_to_save = {}
                for frame in ['gt_frame', 'odom_frame']:
                    dic_to_save.update({frame: {'position': [], 'yaw': []}})
                    for type in ['position', 'yaw']:
                        for i in range(len(data['topics'][tk])):
                            dic_to_save[frame][type].append(data['topics']['goal_pose'][i][frame][type])

                # Write odometry data to a JSON file
                with open(file_path, 'w') as file:
                    json.dump(dic_to_save, file, indent=4)

            elif tk == 'target_object':
                # Filename and path for target object data
                filename = 'traj_target_data'
                file_path = os.path.join(self.dst_datafolder, filename + '.json')

                # Write target object data to a JSON file
                with open(file_path, 'w') as file:
                    json.dump(data['topics'][tk], file, indent=4)

            elif tk in ['rgb', 'depth']:
                # Directory paths for visual data (RGB and Depth images)
                parent_folder_path = os.path.join(self.dst_datafolder, 'visual_data')
                if not os.path.exists(parent_folder_path):
                    os.mkdir(parent_folder_path)  # Create a directory for visual data if it doesn't exist
                
                child_folder_path = os.path.join(parent_folder_path, tk)
                if not os.path.exists(child_folder_path):
                    os.mkdir(child_folder_path)  # Create a subdirectory for the current topic if it doesn't exist
                
                # Save each image in the topic's aligned data
                for i, img in enumerate(data['topics'][tk]):
                    img_name = os.path.join(child_folder_path, f'{i}.jpg')
                    cv2.imwrite(img_name, img)  # Write the image file to disk

            # Log success for each topic
            print(f"[INFO]  {tk} data successfully saved.")


    def save_traj_video(self, data: dict, rate=10):
        """
        Generates and saves a video visualizing trajectory data using aligned data sets.

        This method constructs a video that illustrates the trajectory by plotting positional data over time,
        overlaid on corresponding RGB and depth imagery, and saves it to a designated directory.

        Args:
            data (dict): The aligned data containing trajectories and associated imagery to be visualized.
            rate (int): Frame sampling rate for the video; every 'rate'-th frame is used. Defaults to 10.
        """

        keys = data['topics'].keys()
        images = None
        times = np.array(data['time_elapsed'])
        
        
        for key in keys:
            if key == 'odom':
                # Extract odometry data for positions and orientations
                odom_traj = data['topics'][key]
                positions = np.array([item['odom_frame']['position'] for item in odom_traj])
                yaws = np.array([item['odom_frame']['yaw'] for item in odom_traj])

            if key == 'target_object':
                # Extract target object positions and bounding box dimensions
                target_in_cam = data['topics'][key]
                target_positions = np.array([item.get('position') for item in target_in_cam])
                target_bbox3d = np.array([item.get('bbox3d') for item in target_in_cam])
                target_bbox3d = target_bbox3d[:, :4, :2]  # Use the top view x-y coordinates of the bounding box

            if key == "rgb":
                # Stack RGB images for video frames
                images = np.concatenate((images, data['topics'][key]), axis=1) if images is not None else np.array(data['topics'][key])

            if key == "depth":
                # Convert depth images to RGB format
                depth_images = np.array(data['topics'][key])
                shape = depth_images.shape
                depth_rgb_images = np.zeros((shape[0], shape[1], shape[2], 3), dtype=np.uint8)
                depth_rgb_images[:, :, :, 0] = depth_images  # Red channel
                depth_rgb_images[:, :, :, 1] = depth_images  # Green channel
                depth_rgb_images[:, :, :, 2] = depth_images  # Blue channel
                images = np.concatenate((images, depth_rgb_images), axis=2) if images is not None else depth_rgb_images

        target_positions = np.zeros_like(positions) if "target_object" not in keys else target_positions

        # Setup figure and axes for the trajectory and image plots
        fig = plt.figure(figsize=[16, 12])
        grid = plt.GridSpec(12, 17, hspace=0.2, wspace=0.2)
        ax_image = fig.add_subplot(grid[1:6, :], title="Scene Image")
        ax_trajectory = fig.add_subplot(grid[7:, :], title="Trajectory", xlabel="Y [m]", ylabel="X [m]")

        # Set axis limits based on target and odometry data
        x_lim = np.max([abs(np.max(target_positions[:, 1] + positions[:, 1])), abs(np.min(target_positions[:, 1] + positions[:, 1]))])
        ax_trajectory.set_xlim(xmin=-x_lim - 0.5, xmax=x_lim + 0.5)
        y_lim = np.max([abs(np.max(target_positions[:, 0] + positions[:, 0])), abs(np.min(target_positions[:, 0] + positions[:, 0]))])
        ax_trajectory.set_ylim(ymin=-y_lim - 0.5, ymax=y_lim + 0.5)
        
        ax_trajectory.invert_xaxis()
        ax_trajectory.grid(True)

        aggregated_positions = []
        aggregated_target_positions = []
        Frames = []
        for i in range(len(images)):
            if i % rate == 0:
                # Sample frames according to the specified rate for visualization
                aggregated_positions.append(positions[i])
                corners = None
                if ((target_positions[i] != np.zeros_like(target_positions[i])).all()):
                    # Calculate transformed positions for visualization
                    aggregated_target_positions.append(np.array([[np.cos(yaws[i]), -np.sin(yaws[i])], [np.sin(yaws[i]), np.cos(yaws[i])]]) @ target_positions[i][:2].T + positions[i])
                    corners = (np.array([[np.cos(yaws[i]), -np.sin(yaws[i])], [np.sin(yaws[i]), np.cos(yaws[i])]]) @ target_bbox3d[i].T).T + positions[i]

                frame = TrajViz.visualization(robot_position=np.array(aggregated_positions),
                                            yaw=yaws[i],
                                            curr_image=images[i],
                                            time=times[i],
                                            frame_idx=i,
                                            ax_image=ax_image,
                                            ax_trajectory=ax_trajectory,
                                            target_position=np.array(aggregated_target_positions),
                                            corners=corners)
                Frames.append(frame)

        # Create and save the animation
        ani = animation.ArtistAnimation(fig=fig, artists=Frames, blit=True, interval=200)
        TrajViz.save_animation(ani=ani, dest_dir=self.dst_datafolder, file_name="traj_sample")

    def save_eval_video(self, data: dict, rate=1):
            """
            Generates and saves a video visualizing trajectory data using aligned data sets.

            This method constructs a video that illustrates the trajectory by plotting positional data over time,
            overlaid on corresponding RGB and depth imagery, and saves it to a designated directory.

            Args:
                data (dict): The aligned data containing trajectories and associated imagery to be visualized.
                rate (int): Frame sampling rate for the video; every 'rate'-th frame is used. Defaults to 10.
            """

            keys = data['topics'].keys()
            times = np.array(data['time_elapsed'])
            
            
            for key in keys:
                if key == 'odom':
                    # Extract odometry data for positions and orientations
                    odom_traj = data['topics'][key]
                    positions = np.array([item['odom_frame']['position'] for item in odom_traj])
                    yaws = np.array([item['odom_frame']['yaw'] for item in odom_traj])

                if key == 'goal_pose':
                    # Extract odometry data for positions and orientations
                    goal_pose = data['topics'][key]
                    goal_positions = np.array([item['odom_frame']['position'] for item in goal_pose])
                    goal_yaws = np.array([item['odom_frame']['yaw'] for item in goal_pose])
                
                if key == 'target_object':
                    # Extract target object positions and bounding box dimensions
                    target_in_cam = data['topics'][key]
                    target_positions = np.array([item.get('position') for item in target_in_cam])
                    target_bbox3d = np.array([item.get('bbox3d') for item in target_in_cam])
                    target_bbox3d = target_bbox3d[:, :4, :2]  # Use the top view x-y coordinates of the bounding box

            target_positions = np.zeros_like(positions) if "target_object" not in keys else target_positions
            goal_positions = None if 'goal_pose' not in keys else goal_positions
            goal_yaws = None if 'goal_pose' not in keys else goal_yaws

            # Setup figure and axes for the trajectory and image plots
            fig = plt.figure(figsize=[16, 12])
            grid = plt.GridSpec(12, 17, hspace=0.2, wspace=0.2)
            # ax_image = fig.add_subplot(grid[1:6, :], title="Scene Image")
            ax_trajectory = fig.add_subplot(grid[:, :], title="Trajectory", xlabel="Y [m]", ylabel="X [m]")

            # Set axis limits based on target and odometry data
            x_lim = np.max([abs(np.max(target_positions[:, 1] + positions[:, 1])), abs(np.min(target_positions[:, 1] + positions[:, 1]))])
            ax_trajectory.set_xlim(xmin=-x_lim - 0.5, xmax=x_lim + 0.5)
            y_lim = np.max([abs(np.max(target_positions[:, 0] + positions[:, 0])), abs(np.min(target_positions[:, 0] + positions[:, 0]))])
            ax_trajectory.set_ylim(ymin=-y_lim - 0.5, ymax=y_lim + 0.5)
            
            ax_trajectory.invert_xaxis()
            ax_trajectory.grid(True)

            aggregated_positions = []
            aggregated_target_positions = []
            Frames = []
            for i in range(len(positions)):
                if i % rate == 0:
                    # Sample frames according to the specified rate for visualization
                    aggregated_positions.append(positions[i])
                    corners = None
                    if ((target_positions[i] != np.zeros_like(target_positions[i])).all()):
                        # Calculate transformed positions for visualization
                        aggregated_target_positions.append(np.array([[np.cos(yaws[i]), -np.sin(yaws[i])], [np.sin(yaws[i]), np.cos(yaws[i])]]) @ target_positions[i][:2].T + positions[i])
                        corners = (np.array([[np.cos(yaws[i]), -np.sin(yaws[i])], [np.sin(yaws[i]), np.cos(yaws[i])]]) @ target_bbox3d[i].T).T + positions[i]

                    frame = TrajViz.visualization(robot_position=np.array(aggregated_positions),
                                                yaw=yaws[i],
                                                curr_image=None,
                                                time=times[i],
                                                frame_idx=i,
                                                ax_image=None,
                                                ax_trajectory=ax_trajectory,
                                                target_position=np.array(aggregated_target_positions),
                                                corners=corners,
                                                goal_position=goal_positions[i],
                                                goal_yaw=goal_yaws[i])
                    Frames.append(frame)

            # Create and save the animation
            ani = animation.ArtistAnimation(fig=fig, artists=Frames, blit=True, interval=10)
            TrajViz.save_animation(ani=ani, dest_dir=self.dst_datafolder, file_name="eval_sample")


class BagProcess:
    """
    A class for processing ROS bag files in batches or entire folders.

    This class provides static methods to process ROS bag files from a given folder path. It supports processing
    individual batches of bag files and entire folders containing multiple batches.

    Methods:
        process_batch: Processes a batch of ROS bag files from a specified folder.
        process_folder: Processes multiple batches of ROS bag files from a specified folder.
    """
    
    def __init__(self) -> None:
        pass

    @staticmethod
    def process_batch(bag_folder_path: str,
                    dst_dataset: str = None,
                    dst_datafolder_name: str = None,
                    save_raw: bool = False,
                    save_video: bool = True,
                    config: dict = None,
                    force_process: bool = False,
                    mode: str = "data"):
        """
        Process a batch of ROS bag files located in a specified folder.

        This method reads the metadata from the specified folder and processes each unprocessed bag file found. It can
        optionally save raw data and generate trajectory videos from the processed data.

        Args:
            bag_folder_path (str): Path to the folder containing ROS bag files.
            dst_dataset (str, optional): Destination dataset path. Defaults to None.
            dst_datafolder_name (str, optional): Name of the folder to store processed data. Defaults to None.
            save_raw (bool, optional): Flag to save raw data from the bag files. Defaults to False.
            save_video (bool, optional): Flag to save trajectory videos from the bag files. Defaults to True.
        """

        print(f"[INFO] Batch - {bag_folder_path}")

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

        # Get all filenames in the folder path
        filenames = [filename for filename in os.listdir(bag_folder_path) if filename.endswith(".bag")]
        n_files = len(filenames)
        print(f"[INFO] Found {n_files} bags files")
        
        # Loop through each file in the folder
        bag_i = 1
        for filename in filenames:
                if not(filename in metadata["processed_bags"]) or force_process:
                    # Compress the file using rosbag compress command
                    bagfile = os.path.join(bag_folder_path,filename)
                    
                    print(f"[INFO] Start processing bag [{bag_i} / {n_files}]")
                    bag_reader = BadReader(bagfile=bagfile,
                                        dst_dataset=dst_dataset, 
                                        dst_datafolder_name=dst_datafolder_name,
                                        config=config,
                                        mode = mode)
                    
                    # Get data
                    raw_data = bag_reader.get_raw_dataset()
                    aligned_data = bag_reader.get_aligned_dataset()

                    # Save data
                    if save_raw:
                        bag_reader.save_raw(raw_data)
                    
                    bag_reader.save_aligned(aligned_data)
                    save_video = save_video if config is None else config.get("save_vid", True)
                    
                    if save_video and mode == "data":
                        bag_reader.save_traj_video(aligned_data)

                    if mode == "eval":
                        bag_reader.save_eval_video(aligned_data)

                    # add to metadata
                    metadata["processed_bags"].append(filename)
                else:
                    print(f"[INFO] Bag {filename} already processed")
                
                print("------")
                
                bag_i = bag_i + 1

        # Writing the updated data back to the file
        with open(metadata_file_p, 'w') as file:
            json.dump(metadata, file, indent=4)


    @staticmethod
    def process_folder(folder_path: str =None,
                    dst_dataset: str = None,
                    dst_datafolder_name: str = None,
                    save_raw: bool = False,
                    config:dict = None,
                    force_process: bool = False
                    ):
        
        """
        Process multiple batches of ROS bag files from a specified folder.

        This method iterates through each subfolder within the specified folder, identified as a batch, and processes
        the ROS bag files within using the `process_batch` method.

        Args:
            folder_path (str): Path to the folder containing multiple batches of ROS bag files.
            dst_dataset (str, optional): Destination dataset path. Defaults to None.
            dst_datafolder_name (str, optional): Name of the folder to store processed data. Defaults to None.
            save_raw (bool, optional): Flag to save raw data from the bag files. Defaults to False.
            config (dict, optional): Configuration dictionary containing necessary arguments. Defaults to None.

        """
        

        # Ensure that either config is provided or the other arguments, but not both
        assert (config is None) != (folder_path is None), "Either provide a config dictionary or the individual arguments, but not both"

        if config:
            # Extract arguments from config
            folder_path = config.get('bags_folder')
            dst_dataset = config.get('destination_folder')
            save_raw = config.get('save_raw', True)  # Default to False if not in config
            force_process = config.get('force',False)
            mode = config.get('mode','data')
            
        print(f"[INFO] Processing folder - {folder_path}")
        
        # Loop through each folder
        batches = [batch for batch in os.listdir(folder_path)]
        n_batches = len(batches)
        print(f"[INFO] Found {n_batches} batch folders of bags")
        print("[INFO] ---------- Start Folder Processing -----------")
        batch_i = 1
        for batch in batches:
                batch_path = os.path.join(folder_path, batch)
                record_config_file = os.path.join(batch_path,
                                        "configs",
                                        "record.yaml")

                # Open the record config file
                with open(record_config_file, 'r') as file:
                    record_config = yaml.safe_load(file)
                
                demonstrator = record_config["recording"]["demonstrator"]
                dst_datafolder_name = demonstrator if dst_datafolder_name is None else dst_datafolder_name 
                
                
                print("[INFO] ---------- Start Batch Processing ----------")
                print(f"[INFO] Processing batch [{batch_i} / {n_batches}]")
                BagProcess.process_batch(bag_folder_path=batch_path,
                                        dst_dataset=dst_dataset,
                                        dst_datafolder_name = dst_datafolder_name,
                                        save_raw=save_raw,
                                        config=config.get(demonstrator),
                                        force_process=force_process,
                                        mode=mode)
                print("[INFO] ---------- Finish Batch Processing ----------\n")

                batch_i = batch_i + 1
        print("[INFO] ---------- Finish Folder Processing -----------")


    @staticmethod
    def reset_batch(bag_folder_path: str):
        # Define the file name we're looking for
        file_name = "metadata.json"
        
        # Construct the full file path
        file_path = os.path.join(bag_folder_path, file_name)
        
        # Check if the file exists in the given folder
        if os.path.isfile(file_path):
            # If it exists, call the reset_processed_bags function
            reset_processed_bags(file_path)
        else:
            print(f"{file_name} not found in {bag_folder_path}")

    @staticmethod
    def reset_folder(parent_folder_path: str):
        # List all entries in the parent folder
        for entry in os.listdir(parent_folder_path):
            # Construct full path to the entry
            entry_path = os.path.join(parent_folder_path, entry)
            
            # Check if the entry is a directory (folder)
            if os.path.isdir(entry_path):
                # Apply the process_metadata_in_folder function to the folder
                BagProcess.reset_batch(entry_path)
            else:
                print(f"{entry_path} is not a folder, skipping...")

def main():

    bp = BagProcess()

    config_path = 'process_bag_config.yaml'
    with open(config_path, 'r') as file:
            process_config = yaml.safe_load(file)
    
    bp.process_folder(config=process_config)

if __name__ == "__main__":
    main()