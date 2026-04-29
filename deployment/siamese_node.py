#!/usr/bin/env python

import rospy
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image as PILImage
from sensor_msgs.msg import Image
from std_msgs.msg import Int8

# Import your custom network class and transforms
from siamese_network import SiameseLossDetector
from dataset_utils import get_transforms 

class PayloadLossDetector:
    def __init__(self):
        rospy.init_node('payload_loss_detector', anonymous=True)
        
        self.device = torch.device("cpu") # Advantech MIO-5393
        
        # Load the PyTorch Model
        model_path = rospy.get_param('~model_path', 'siamese_loss.pth')
        self.model = SiameseLossDetector().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((120, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Caching logic
        self.is_initialized = False
        self.cached_embeddings = {0: None, 1: None, 2: None, 3: None}
        self.latest_frames = {0: None, 1: None, 2: None, 3: None}
        
        # Round-robin tracker
        self.current_cam_idx = 0
        
        # ROS Subscribers
        rospy.Subscriber("/cam_front/image_raw", Image, self.cam_callback, callback_args=0)
        rospy.Subscriber("/cam_back/image_raw", Image, self.cam_callback, callback_args=1)
        rospy.Subscriber("/cam_left/image_raw", Image, self.cam_callback, callback_args=2)
        rospy.Subscriber("/cam_right/image_raw", Image, self.cam_callback, callback_args=3)
        
        # ROS Publisher (Outputs 0 for Present, 1 for Loss)
        self.alert_pub = rospy.Publisher("/payload/loss_detected", Int8, queue_size=10)
        
        # Inference Timer (e.g., 10 Hz means it checks 1 camera every 0.1 seconds. 
        # All 4 cameras are checked every 0.4 seconds)
        rospy.Timer(rospy.Duration(0.1), self.inference_loop)
        
        rospy.loginfo("Payload Loss Node Started. Waiting for all 4 cameras...")

        # Alert logic and debouncing
        self.loss_counter = 0
        self.emergency_stop_triggered = False

    def cam_callback(self, msg, cam_id):
        try:
            # Decode the raw bytes directly into a numpy array
            img_array = np.frombuffer(msg.data, dtype=np.uint8)
            
            # Reshape it using the dimensions provided by the camera
            cv_image = img_array.reshape((msg.height, msg.width, 3))
            
            # cv_camera usually outputs BGR. PyTorch expects RGB.
            if msg.encoding == 'bgr8':
                cv_image = cv_image[:, :, ::-1] # Quick BGR to RGB swap
                
            self.latest_frames[cam_id] = cv_image
        except Exception as e:
            rospy.logerr(f"Camera {cam_id} decode error: {e}")

    def cache_initial_state(self):
        """Runs once to generate the anchor embeddings."""
        if any(frame is None for frame in self.latest_frames.values()):
            return # Waiting for all cameras to publish at least one frame
            
        rospy.loginfo("All cameras online. Waiting 1.5 seconds for camera auto-exposure to settle...")
        rospy.sleep(1.5)
        
        rospy.loginfo("Caching initial payload state...")
        
        with torch.no_grad():
            for cam_id in range(4):
                frame = self.latest_frames[cam_id]
                # Convert NumPy array to PIL Image for PyTorch
                frame_pil = PILImage.fromarray(frame)
                tensor = self.transform(frame_pil).unsqueeze(0).to(self.device)

                # Assume your model has a method to just run the encoder
                embedding = self.model.forward_one(tensor)
                self.cached_embeddings[cam_id] = embedding

        self.is_initialized = True
        rospy.loginfo("Initial state cached. Commencing active monitoring.")

    def inference_loop(self, event):
        if self.emergency_stop_triggered:
            return

        rospy.loginfo(f"Inference loop started on Camera {self.current_cam_idx}")

        """The round-robin inference engine."""
        if not self.is_initialized:
            self.cache_initial_state()
            return

        # 1. Grab the latest frame for the current camera in the round-robin queue
        frame = self.latest_frames[self.current_cam_idx]
        if frame is None:
            return

        # 2. Preprocess
        frame_pil = PILImage.fromarray(frame)
        tensor = self.transform(frame_pil).unsqueeze(0).to(self.device)

        # 3. Inference against the specific cached embedding
        with torch.no_grad():
            current_embedding = self.model.forward_one(tensor)
            anchor_embedding = self.cached_embeddings[self.current_cam_idx]

            # Assume forward_mlp takes the two embeddings and applies the absolute difference + MLP
            prediction = self.model.forward_mlp(anchor_embedding, current_embedding)

            # Threshold the Sigmoid output
            prob = prediction.item()
            loss_detected = 1 if prediction.item() > 0.5 else 0

        status_text = "LOSS" if loss_detected else "NORMAL"
        rospy.loginfo(f"Cam {self.current_cam_idx} | Score: {prob:.4f} | Status: {status_text}")

        # 4. Publish alert if payload is missing
        if loss_detected == 1:
            self.loss_counter += 1
            rospy.logwarn(f"PAYLOAD LOSS DETECTED {self.loss_counter} on Camera {self.current_cam_idx}!")
        else:
            self.loss_counter = 0

        if self.loss_counter >= 3:
            rospy.logwarn("*** 3 CONSECUTIVE LOSSES DETECTED. STOPPING. ***")
            self.alert_pub.publish(1)
            self.emergency_stop_triggered = True
        else:
            self.alert_pub.publish(0)

        # 5. Advance the round-robin index
        self.current_cam_idx = (self.current_cam_idx + 1) % 4

if __name__ == '__main__':
    try:
        detector = PayloadLossDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
