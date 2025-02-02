# bagtool: ROS Bag Management for Dataset Collection

--- 

## Overview

TBD...

---

## Installation

### Prerequisites

- [Ubuntu 20.04 (Focal Fossa)](https://releases.ubuntu.com/focal/)
- [ROS Noetic](https://wiki.ros.org/noetic/Installation/Ubuntu)
- [zed-ros-wrapper (v4.1.0 release or newer)](https://github.com/stereolabs/zed-ros-wrapper) 


### Build the package

Clone the repo:

```bash
$ git clone https://github.com/nimiCurtis/bagtool
```

Install the dependencies by:

```bash
$ pip install -r requirements.txt 
```

Build the package:
```bash
$ pip install -e . 
```

---

## Usage

**Important: This package is designed to work with bag files structured in accordance with the recording conventions established by the [zion_zed_ros_interface](https://github.com/nimiCurtis/zion_zed_ros_interface). Please ensure compatibility with this structure for optimal performance. for more info on the structure use the given link**


### Main Tool - Process

The main tool of this package is the process tool, which process ROS bag files from a given folder path and saving the data in useful formats and structures for future works.

#### You can use the tool in two ways:

- From the CLI : 



<!-- #### Using from the command-line -->
    bagtool process [-h] [-b BATCH] [-f FOLDER] [-d DST] [-n NAME]
                        [--no_raw]

    optional arguments:
    -h, --help            show this help message and exit
    -b BATCH, --batch BATCH
                            path to a bag batch folder
    -f FOLDER, --folder FOLDER
                            path to a bag folder consisting bag batches
    -d DST, --dst DST     path to a dataset destination folder
    -n NAME, --name NAME  a custom name for the datafolder
    --no_raw              disable raw data saveing


- And From a custom script using the bagtool.process module 

#### Example usecase:

Assuming you have a folder with a bag batch folder (see the *important* note) at `<some_path>/bag` folder. And you'd like to set your entire dataset on `<some_path>/dataset`. 

**Using CLI:**
<!-- #### Using from the command-line -->
    bagtool process -f <some_path>/bag --dst <some_path>/dataset

**Using custom script:**
```python
from bagtool.process.process import BagProcess as bp

source = '<some_path>/bag'
dest = '<some_path>/dataset'
bp.process_folder(source,dest)
```

**(Recomended) Using custom script + .yaml config:**

We can a extract the dataset from a recordings database using custom config file.

Such a config would looks as follow:

```yaml
bags_folder: "/home/roblab20/catkin_ws/src/zion_ros/zion_zed_ros_interface/bag"
destination_folder: "/home/roblab20/dev/bagtool/dataset"
save_raw: true                        # I dont see a reason why to set it to false

demonstrator_name:                    # demonstrator/robot name
  aligned_topics: ["odom", "rgb", "depth", <topic key name>]     # pick topics to be aligned
  sync_rate: 20                       # alignment rate, if null we will 
                                      # use the min frequency, from the aligned topics frequency
  save_vid: true                      # saving video
  ## more params .. 
```

Process file would looks as follow:
```python
import yaml
from bagtool.process.process import BagProcess as bp

def main():
    
    config_path = 'process_bag_config.yaml'
    with open(config_path, 'r') as file:
            process_config = yaml.safe_load(file)
    
    bp.process_folder(config=process_config)

if __name__ == "__main__":
    main()
```



**Dataset structure:**

The resulted dataset from the example above will be as follow:


```
├── dataset
│   ├── <name_of_bag_batch1>
|   |    ├── raw_data
│   │       ├── raw_<topic1>.h5
│   │       ├── ...
│   │       └── raw_<topicN>.h5
|   |    ├── visual_data
|   |       ├── depth            
│   │           ├── 0.jpg
│   │           ├── ...
│   │           └── T.jpg
|   |       └── rgb          
│   │           ├── ...
│   │           └── T.jpg
│   │    ├── metadata.json
│   │    ├── robot_traj_data.json
│   │    ├── target_traj_data.json (when using the object detection topic)
│   │    └── traj_sample.mp4
│   ...
└── └── <name_of_bag_batchN>
         ├── raw_data
            ├── raw_<topic1>.h5
            ├── ...
            └── raw_<topicN>.h5
         ├── visual_data
             ├── depth            
                 ├── 0.jpg
                 ├── ...
                 └── T.jpg
             └── rgb          
                 ├── ...
                 └── T.jpg
         ├── metadata.json
         ├── robot_traj_data.json
         ├── target_traj_data.json (when using the object detection topic)
         └── traj_sample.mp4
```  

### Secondary Tool - Compress
Another tool provided is a quick & simple command line interface for rosbag compressing of entire batches of bags and folder of batches.


- (Recomended) From the CLI : 

<!-- #### Using from the command-line -->
    usage: bagtool compress [-h] [-b BATCH] [-f FOLDER]

    optional arguments:
    -h, --help            show this help message and exit
    -b BATCH, --batch BATCH
                            path to a bag batch folder
    -f FOLDER, --folder FOLDER
                            path to a bag folder consisting bag batches


---

### TODO: 
By priority Top-Down:
- [ ] Improve efficiency and modularity
- [ ] Handling with 32bit depth images ? 
- [ ] Check use of the compressed Image sensor msgs
- [ ] Add arrow to the traj animation indicating the yaw direction ?
