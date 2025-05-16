# Cable Routing

This directory contains the core components for robotic cable routing, including planning algorithms, environment setup, perception, and learned models.

## Structure

- **`algo/`**: Planning algorithms (e.g. A* search, Levenshtein distance).
- **`configs/`**: Configuration files for boards, cameras, and environment setup.
- **`env/`**: Interfaces for cameras, robots, and the environment, with ROS integration.
- **`handloom/`**: Autoregressive cable tracing.
- **`scripts/`**: End-to-end pipeline scripts, calibration, data collection, and demos.

---

## Installation

To set up the environment, install dependencies, and build required components, please refer to the [INSTALL.md](/INSTALL.md) guide.

---

## Usage Guide

### 🔧 Step 1: Launch the Robot and Camera Display

1. Turn on the YuMi robot and power supply.
2. Open Terminal 1: Launch the camera node
   ```bash
   mamba activate cable
   yumi  # source the ROS workspace
   roslaunch zed_wrapper zedm.launch
   ```
3. Open Terminal 2: Open visualization
   ```bash
   mamba activate cable
   yumi  # source the ROS workspace
   rviz
   ```

---

### 🧭 Step 2: Create the Board Configuration

1. Launch the board setup GUI:
   ```bash
   cd ~/cable_routing/cable_routing/env/board/
   python board_setup_gui.py
   ```
2. Follow the on-screen instructions (start with point 'A').
3. Press 'q' to save the board layout and validate the generated image.

---

### 🤖 Step 3: Run the Cable Routing Pipeline

1. Open the routing script:
   ```bash
   ~/cable_routing/cable_routing/scripts/cable_routing_pipeline.py
   ```
2. Inside the script, define the desired routing sequence:
   ```python
   routing = ["A", "B", "C", "E", "F"]
   ```
3. Execute the pipeline:
   ```bash
   python ~/cable_routing/cable_routing/scripts/cable_routing_pipeline.py
   ```

---
