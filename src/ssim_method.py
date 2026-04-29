from pathlib import Path
import time
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from .utils import iter_video_frames, IGNORE_FIRST_N_FRAMES

class SSIMMethod:
    def __init__(self, threshold=0.85, consecutive_frames=5):
        self.threshold = threshold
        self.consecutive_frames = consecutive_frames
        self.reference_frame = None

    def preprocess_frame(self, frame):
        # Apply standard noise reduction and preprocessing
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        return blur

    def process_frame(self, frame):
        start_time = time.perf_counter()
        frame = self.preprocess_frame(frame)
        
        # Compute structural similarity
        if self.reference_frame is None:
            # Baseline setting
            self.reference_frame = frame
            score = 1.0 # 1.0 indicates perfect structural similarity
        else:
            if frame.shape != self.reference_frame.shape:
                frame = cv2.resize(
                    frame,
                    (self.reference_frame.shape[1], self.reference_frame.shape[0])
                )

            # SSIM returns a structural similarity index [-1, 1], where 1 is identical
            score, _ = ssim(
                self.reference_frame,
                frame,
                full=True,
                data_range=255
            )

            score = max(0.0, float(score)) # clamp to strictly 0 or positive for simplicity
            
        inference_time_ms = (time.perf_counter() - start_time) * 1000.0
        
        return score, inference_time_ms

    def predict_video(self, video_path: Path):
        self.reset()

        scores = []
        latencies_ms = []
        frame_predictions = []

        consecutive_count = 0
        detected_frame = -1

        for frame_idx, frame in iter_video_frames(video_path):
            score, inference_time_ms = self.process_frame(frame)

            scores.append(score)
            latencies_ms.append(inference_time_ms)

            if frame_idx <= IGNORE_FIRST_N_FRAMES:
                pred_loss = 0
            else:
                raw_loss = score < self.threshold
                if raw_loss:
                    consecutive_count += 1
                else:
                    consecutive_count = 0

                pred_loss = 1 if consecutive_count >= self.consecutive_frames else 0

            frame_predictions.append(pred_loss)

            if pred_loss == 1 and detected_frame == -1:
                detected_frame = frame_idx

        return {
            "detected_frame": detected_frame,
            "frame_predictions": frame_predictions,
            "scores": scores,
            "latencies_ms": latencies_ms,
        }

    def reset(self):
        """Reset the baseline frame reference for a new video."""
        self.reference_frame = None