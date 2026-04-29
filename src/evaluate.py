import pandas as pd

from .utils import GROUND_TRUTH_CSV, resolve_test_video_path, IGNORE_FIRST_N_FRAMES
from .metrics import summarize_metrics


def evaluate_method(method, ground_truth_csv=GROUND_TRUTH_CSV):
    """
    Evaluates one method on all videos in ground_truth.csv.

    Each method must implement:
        predict_video(video_path: Path) -> dict

    Expected predict_video output:
        {
            "detected_frame": int,              # -1 if no loss detected
            "frame_predictions": list[int],     # 0/1 per frame
            "scores": list[float],              # score per frame
            "latencies_ms": list[float],        # latency per frame
        }
    """

    gt = pd.read_csv(ground_truth_csv)

    all_video_results = []
    all_frame_results = []

    for _, row in gt.iterrows():
        video_path = resolve_test_video_path(row["filename"])
        result = method.predict_video(video_path)

        detected_frame = result["detected_frame"]
        frame_predictions = result["frame_predictions"]
        scores = result["scores"]
        latencies_ms = result["latencies_ms"]

        video_result = {
            "filename": row["filename"],
            "camera_id": row["camera_id"],
            "is_loss_event": row["is_loss_event"],
            "loss_frame": row["loss_frame"],
            "total_frames": row["total_frames"],
            "detected_frame": detected_frame,
        }

        all_video_results.append(video_result)

        for i, pred in enumerate(frame_predictions):
            frame_idx = i + 1

            if frame_idx <= IGNORE_FIRST_N_FRAMES:
                pred = 0

            all_frame_results.append({
                "filename": row["filename"],
                "camera_id": row["camera_id"],
                "is_loss_event": row["is_loss_event"],
                "loss_frame": row["loss_frame"],
                "frame_idx": frame_idx,
                "score": scores[i] if i < len(scores) else None,
                "pred_frame_loss": pred,
                "inference_time_ms": latencies_ms[i] if i < len(latencies_ms) else None,
            })

    video_results = pd.DataFrame(all_video_results)
    frame_results = pd.DataFrame(all_frame_results)

    metrics = summarize_metrics(video_results, frame_results)

    return video_results, frame_results, metrics