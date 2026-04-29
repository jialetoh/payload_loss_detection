import pandas as pd
from .utils import IGNORE_FIRST_N_FRAMES, EARLY_TOLERANCE_FRAMES

def add_frame_ground_truth(frame_results: pd.DataFrame) -> pd.DataFrame:
    """
    Adds gt_frame_loss:
    - 1 if loss has occurred at this frame
    - 0 otherwise
    """
    df = frame_results.copy()

    df["gt_frame_loss"] = (
        (df["is_loss_event"] == 1) &
        (df["frame_idx"] >= (df["loss_frame"] - EARLY_TOLERANCE_FRAMES)) &
        (df["frame_idx"] > IGNORE_FIRST_N_FRAMES)
    ).astype(int)

    return df


def event_level_recall(video_results: pd.DataFrame) -> float:
    loss_videos = video_results[video_results["is_loss_event"] == 1]

    if len(loss_videos) == 0:
        return 0.0

    correct = (
        (loss_videos["detected_frame"] != -1) &
        (loss_videos["detected_frame"] >= (loss_videos["loss_frame"] - EARLY_TOLERANCE_FRAMES))
    ).sum()

    return correct / len(loss_videos)


def event_level_precision(video_results: pd.DataFrame) -> float:
    detected = video_results[video_results["detected_frame"] != -1]

    if len(detected) == 0:
        return 0.0

    correct = (
        (detected["is_loss_event"] == 1) &
        (detected["detected_frame"] >= (detected["loss_frame"] - EARLY_TOLERANCE_FRAMES))
    ).sum()

    return correct / len(detected)


def frame_level_precision(frame_results: pd.DataFrame) -> float:
    df = add_frame_ground_truth(frame_results)

    tp = ((df["pred_frame_loss"] == 1) & (df["gt_frame_loss"] == 1)).sum()
    fp = ((df["pred_frame_loss"] == 1) & (df["gt_frame_loss"] == 0)).sum()

    if tp + fp == 0:
        return 0.0

    return tp / (tp + fp)


def average_detection_delay(video_results: pd.DataFrame):
    valid = video_results[
        (video_results["is_loss_event"] == 1) &
        (video_results["detected_frame"] != -1) &
        (video_results["detected_frame"] >= (video_results["loss_frame"] - EARLY_TOLERANCE_FRAMES))
    ].copy()

    if len(valid) == 0:
        return None

    valid["delay_frames"] = valid["detected_frame"] - valid["loss_frame"]
    return valid["delay_frames"].mean()


def average_latency_ms(frame_results: pd.DataFrame) -> float:
    return frame_results["inference_time_ms"].mean()


def summarize_metrics(video_results: pd.DataFrame, frame_results: pd.DataFrame) -> dict:
    return {
        "event_level_recall": event_level_recall(video_results),
        "event_level_precision": event_level_precision(video_results),
        "frame_level_precision": frame_level_precision(frame_results),
        "avg_detection_delay_frames": average_detection_delay(video_results),
        "avg_latency_ms": average_latency_ms(frame_results),
    }