# ARTE Payload Loss Detection Pipeline

This directory contains the payload loss detection pipeline designed for **ARTE** (Autonomous Robot for Transporting Equipment). The system uses four cameras and a trained Machine Learning model (siamese_loss.pth) to visually monitor the payload in real-time. If the payload is removed or missing, it triggers an emergency stop and plays 3 beeps to alert the user.

The system has not been integrated into the main navigation or UI notification code due to time constraints and is currently just a proof of concept. Running the demo makes the robot drive straight for 5 meters and stop if the payload loss is detected.

---

## What Does Each File Do?

### Neural Network (Machine Learning)
* **`siamese_network.py`**: Defines the architecture of the AI model.
* **`siamese_loss.pth`**: The trained model weights. This is the compiled "knowledge" of what the payload looks like.
* **`dataset_utils.py`**: Helper functions that format images (resizing, recoloring) before feeding them into the AI model.

### Robot Integration (ROS)
* **`siamese_node.py`**: The core AI application running on the robot. It hooks into ARTE's four camera feeds, runs inference against the AI model smoothly in a round-robin schedule (so the computer doesn't lag), and actively broadcasts an alarm on the `/payload/loss_detected` topic if the payload is lost.
* **`demo_controller.py`**: A standalone script designed specifically for testing. It instructs the bot to drive straight at a fixed speed for a set distance. It listens to `/payload/loss_detected`, executing an emergency stop and playing 3 beeps if something goes wrong.
* **`run_demo.sh`**: A master launcher script. Running this single file automatically starts the underlying ROS services, the 4 cameras, the AI tracking node, and the demo driving behavior in the correct order.

---

## Running the Demo

To test the payload detection pipeline on the physical robot:

1. Ensure that there is more than 5 meters of clear space in front of the robot.
2. Place the payload on the bot.
3. Navigate to this directory and run the launcher:
   ```bash
   cd ~/Desktop/payload_loss_recordings
   ./run_demo.sh
   ```
4. **What happens**: The system will initialize (takes about 10 seconds) and the Siamese node will automatically take a snapshot (won't save the picture but its mathematical features will be extracted) from all four cameras to serve as "reference anchors" of the payload from every angle.
5. The bot will begin driving straight for 5 meters. If the payload is removed during the movement, the bot will stop and give 3 warning beeps.

---

## Future Integration into ARTE's Main Navigation

Right now, `demo_controller.py` is being used to tell the robot when to stop. In the future, once ARTE is navigating autonomously around a building, you will no longer need `demo_controller.py`. 

Instead, you can directly patch the payload loss detector into ARTE's main autonomous navigation script (e.g. your `move_base` interface) using the following steps:

1. **Launch the Detector**: Ensure `siamese_node.py` is launched alongside the cameras in your main launch file. 
2. **Setup a Listener**: In your main navigation controller (usually written in Python or C++), subscribe to the ROS topic `/payload/loss_detected`.
3. **Handle an Alert**: The Siamese node broadcasts a `1` when a loss is confidently detected. When your navigation code sees this `1`:
   * Instantly cancel the current path-finding goal.
   * Send a zero-velocity message (`Twist` populated with `0`s) to the robot velocity controller (`cmd_vel`) to slam the brakes.
   * Play any required warning alarms or trigger staff notification systems.

---

## Data Privacy & Security

Due to the highly sensitive environment and strict security constraints, **no images are ever stored on disk**. 
The `siamese_node.py` and `siamese_network` operate purely in RAM:
* When the robot initializes, it captures a "reference anchor" of the payload. 
* Instead of saving this image, the neural network extracts a mathematical representation of the image's "features" (an embedding) and saves *only* these features in RAM.
* During active monitoring, live frames are compared against these mathematical features to verify payload presence. Live images are processed and instantly discarded.

---

## Known Issues & Future Work

As this is currently a proof-of-concept, there are a few unresolved bugs that require further testing:
* **False Negatives (Dataset Limitation)**: The network may occasionally miss payload loss events during operation. The current model was trained on a limited initial dataset to establish this proof-of-concept. To improve robustness and generalization, the model must be retrained using a larger, more comprehensive dataset captured within the intended operating environment.
* **USB Bandwidth Constraints**: All four cameras currently route through a single USB hub connected to one port on the robot's computer. This configuration saturates the USB bandwidth limit, which can prevent all four cameras from streaming simultaneously even at reduced resolutions. To alleviate this throughput bottleneck, cameras should be distributed across multiple independent USB ports on the bot.
