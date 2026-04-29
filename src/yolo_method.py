from pathlib import Path
import time

from ultralytics import YOLO

from .utils import iter_video_frames, IGNORE_FIRST_N_FRAMES, WEIGHTS_DIR


class YOLOMethod:
    def __init__(
        self,
        weights_path: Path | None = None,
        conf_threshold=0.25,
        count_drop_threshold=1,
        consecutive_frames=5,
    ):
        self.weights_path = weights_path or (WEIGHTS_DIR / "yolo26n.pt")
        self.conf_threshold = conf_threshold
        self.count_drop_threshold = count_drop_threshold
        self.consecutive_frames = consecutive_frames

        self.model = YOLO(str(self.weights_path))
        self.initial_count = None

    def count_objects(self, frame):
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            verbose=False,
        )

        boxes = results[0].boxes
        return len(boxes) if boxes is not None else 0

    def process_frame(self, frame):
        start_time = time.perf_counter()

        object_count = self.count_objects(frame)

        if self.initial_count is None:
            self.initial_count = object_count

        count_drop = self.initial_count - object_count
        score = float(count_drop)

        inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        return score, object_count, inference_time_ms

    def predict_video(self, video_path: Path):
        self.reset()

        scores = []
        latencies_ms = []
        frame_predictions = []
        object_counts = []

        consecutive_count = 0
        detected_frame = -1

        for frame_idx, frame in iter_video_frames(video_path):
            score, object_count, inference_time_ms = self.process_frame(frame)

            scores.append(score)
            object_counts.append(object_count)
            latencies_ms.append(inference_time_ms)

            if frame_idx <= IGNORE_FIRST_N_FRAMES:
                pred_loss = 0
            else:
                raw_loss = score >= self.count_drop_threshold

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
            "object_counts": object_counts,
        }

    def reset(self):
        self.initial_count = None


        