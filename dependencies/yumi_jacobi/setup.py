"""
Setup of YuMi RWS python codebase
Author: Mike Danielczuk
"""
import os
from setuptools import setup, find_packages

root_dir = os.path.dirname(os.path.realpath(__file__))

requirements = [
    "numpy",
    "autolab-core",
    "jacobi-motion"]

# load __version__ without importing anything
version_file = os.path.join(os.path.dirname(__file__), "yumi_jacobi/version.py")
with open(version_file, "r") as f:
    # use eval to get a clean string of version from file
    __version__ = eval(f.read().strip().split("=")[-1])

# load README.md as long_description
long_description = ""
if os.path.exists("README.md"):
    with open("README.md", "r") as f:
        long_description = f.read()

setup(
    name="yumi_jacobi",
    version=__version__,
    description="YuMi with Jacobi Robotics Motion Planning thru EGM and Robot Web Services (RWS) gripper control for YuMi smart grippers",
    long_description=long_description,
    author="Justin Yu",
    author_email="yujustin@berkeley.edu",
    license="MIT Software License",
    url="https://github.com/uynitsuj/yumi_jacobi",
    keywords="robotics grasping",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
    ],
    packages=find_packages(),
    install_requires=requirements,
)