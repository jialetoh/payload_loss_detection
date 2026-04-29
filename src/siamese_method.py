from pathlib import Path
import sys
import time

import cv2
import torch
from PIL import Image
import torchvision.transforms as transforms

from .utils import iter_video_frames, IGNORE_FIRST_N_FRAMES, WEIGHTS_DIR, PROJECT_ROOT

SIAMESE_DIR = PROJECT_ROOT / "siamese"
sys.path.append(str(SIAMESE_DIR))

from siamese_model import SiameseLossDetector


class SiameseMethod:
    def __init__(
        self,
        weights_path: Path | None = None,
        threshold=0.5,
        consecutive_frames=5,
        device="cpu",
    ):
        self.weights_path = weights_path or (WEIGHTS_DIR / "siamese_best.pth")
        self.threshold = threshold
        self.consecutive_frames = consecutive_frames
        self.device = torch.device(device)

        self.model = SiameseLossDetector().to(self.device)
        self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((120, 160)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.reference_embedding = None

    def preprocess_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        tensor = self.transform(frame_pil).unsqueeze(0).to(self.device)
        return tensor

    def process_frame(self, frame):
        start_time = time.perf_counter()

        tensor = self.preprocess_frame(frame)

        with torch.no_grad():
            current_embedding = self.model.forward_one(tensor)

            if self.reference_embedding is None:
                self.reference_embedding = current_embedding
                score = 0.0
            else:
                prediction = self.model.forward_mlp(
                    self.reference_embedding,
                    current_embedding,
                )
                score = float(prediction.item())

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
                raw_loss = score >= self.threshold

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
        self.reference_embedding = None

        