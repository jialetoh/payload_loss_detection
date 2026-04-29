#!/bin/bash

# 1. Trap Ctrl+C to ensure all background ROS nodes are killed when you exit
trap "kill -- -$$" SIGINT SIGTERM EXIT

echo "[INFO] Starting roscore..."
roscore &
sleep 2

echo "[INFO] Starting scout base..."
roslaunch scout_base scout_base.launch &
sleep 2

echo "[INFO] Activating Conda environment 'payload_cv'..."
# Adjust the path below if your conda is installed in ~/anaconda3 instead of miniconda3
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate payload_cv

echo "[INFO] Starting USB Camera Node..."
# Change /dev/video0 if your camera is mounted on a different port
rosrun cv_camera cv_camera_node __name:=cam_right _device_id:=0 _image_width:=160 _image_height:=120 _rate:=10 &
rosrun cv_camera cv_camera_node __name:=cam_front _device_id:=2 _image_width:=160 _image_height:=120 _rate:=10 &
rosrun cv_camera cv_camera_node __name:=cam_back _device_id:=4 _image_width:=160 _image_height:=120 _rate:=10 &
rosrun cv_camera cv_camera_node __name:=cam_left _device_id:=6 _image_width:=160 _image_height:=120 _rate:=10 &
sleep 2 # Give the camera hardware a few seconds to warm up and publish

echo "[INFO] Starting Siamese Inference Node..."
# Run the node in the background. It will automatically cache the initial frame.
python siamese_node.py &
sleep 4 # Wait 5 seconds for PyTorch to load and the anchor image to be cached

echo "[INFO] Starting Bot Movement (Demo Controller)..."
echo "[WARNING] The bot will now move forward 5 meters!"
# Run the movement script in the foreground
python demo_controller.py --distance 5.0 --speed 0.3

# The foreground process has finished, the script will now safely exit and trigger the trap
