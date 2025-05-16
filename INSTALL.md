# Installation

> 💡 **Note:** Currently, we use ROS only for the camera module. This dependency can be easily replaced. However, if you plan to perform real-time control, following all installation steps is necessary.

(From Justin's repo)

Full install tested on Ubuntu 22.04 ROS Noetic in mamba-forge environment.


```
cd ~/
git clone --recurse-submodules https://github.com/osheraz/cable_routing.git
```

## Install Mamba-Forge
Install mamba-forge
Download the installer using curl or wget or your favorite program and run the script.
For eg:
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```
or
```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

## Create Mamba-Forge environment
```bash
mamba create -n cable python=3.11
mamba activate cable
```

```bash
# this adds the conda-forge channel to the new created environment configuration 
conda config --env --add channels conda-forge
# and the robostack channel
conda config --env --add channels robostack-staging
# remove the defaults channel just in case, this might return an error if it is not in the list which is ok
conda config --env --remove channels defaults
```

## Install 
```bash
cd ~/cable_routing
python -m pip install -e .
```

## Install ROS Noetic into the environment
```bash
mamba install ros-noetic-desktop
mamba deactivate
mamba activate cable

mamba install compilers cmake pkg-config make ninja colcon-common-extensions catkin_tools rosdep
```

### Test mamba-forge ros installation
```bash
mamba activate cable
roscore
```
Should start a roscore instance

## Install ZED Drivers, Jacobi, and HANDLOOM

Follow the setup instructions for each component:

- 📦 **Jacobi**: Refer to the `README.md` inside the `yumi_jacobi` directory.
- 🧶 **HANDLOOM**: Refer to the `README.md` inside the `handloom` directory.
- 🎥 **ZED Drivers**: Clone the ZED ROS wrapper and compile it in your `catkin_ws`.

  ```bash
  cd ~/catkin_ws/src
  git clone https://github.com/stereolabs/zed-ros-wrapper.git
  cd ..
  catkin_make
  ```
