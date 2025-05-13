# ABB YuMi IRB 14000 Dual-Arm Robot Jacobi Motion Wrapper

This YuMi wapper uses the **Jacobi Motion Library** for robot motion planning and Externally Guided Motion (EGM) for low level robot hardware control. Jacobi Motion provides a clean API for efficient algorithms that compute time-optimized, jerk-limited robot arm trajectories in milliseconds.

Primary reasons for the wrapper are to 1. Allow AUTOLAB-Core `RigidTransform` objects to define cartesian motion waypoints and goals and 2. include YuMi IRB 14000 [SmartGripper](https://library.e.abb.com/public/6c35d74e5be34fcb93bf6810c449a2bd/3HAC054949%20PM%20IRB%2014000%20Gripper-en.pdf) functionality and interface which is not natively supported through Jacobi at this time. Custom RAPID Modules for the Smartgrippers were uploaded onto the YuMi, so for any inquiries about adding SmartGripper support to Jacobi Modules + YuMi, please direct email inquiries to yujustin@berkeley.edu. All motion planning functionality is taken care of by the Jacobi interface.

For instructions regarding installation, getting started, and for general documentation refer to [docs.jacobirobotics.com](https://docs.jacobirobotics.com).

## Installation
The Jacobi Python package and YuMi drivers can be installed from PyPI via

```
pip install jacobi-motion
pip install jacobi-abb-driver
```

Currently tested up to `jacobi-motion==0.0.43` and `jacobi-abb-driver==0.0.40` though later versions may still work depending on support and code compatibility.

In case no package could be found, make sure to upgrade pip to the latest available version via pip install pip --upgrade.

Alternatively, go to the downloads page for [Jacobi Motion](https://account.jacobirobotics.com/downloads/jacobi-motion) and [Jacobi Driver](https://account.jacobirobotics.com/downloads/jacobi-drivers) and download the appropriate Python wheels and ABB robot drivers manually.

Then install this package with:
```
pip install -e .
```
which will allow you to simply import the robot wrapper interface with:
```
from yumi_jacobi.interface import Interface
```

## Getting Started
Refer to the [test_interface.py](https://github.com/uynitsuj/yumi_jacobi/blob/master/starter_examples/test_interface.py) file for an example on how to use the wrapper interface for the YuMi robot. Also refer the [Jacobi Documentation](https://docs.jacobirobotics.com) as necessary as the `Interface` class contains two `YuMiArm` robot driver classes (one per arm) which inherit from Jacobi's [ABBDriver](https://docs.jacobirobotics.com/api/robot_driver_py.html#ABBDriver) class.

## Jacobi License
The Cloud version requires an API key for authentication that you can access at your user account at [account.jacobirobotics.com](account.jacobirobotics.com).
