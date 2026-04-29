from pathlib import Path
import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DATA_DIR = PROJECT_ROOT / "test_data"
GROUND_TRUTH_CSV = TEST_DATA_DIR / "ground_truth.csv"
WEIGHTS_DIR = PROJECT_ROOT / "weights"

IGNORE_FIRST_N_FRAMES = 30  # Skip first n frames to allow camera feed to warm up
EARLY_TOLERANCE_FRAMES = 60 # Detection tolerance window

def resolve_test_video_path(filename: str) -> Path:
    return TEST_DATA_DIR / filename


def open_video(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    return cap


def iter_video_frames(video_path: Path):
    """
    Yield (frame_idx, frame), where frame_idx starts from 1.
    """
    cap = open_video(video_path)
    frame_idx = 1

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        yield frame_idx, frame
        frame_idx += 1

    cap.release()


def ensure_parent_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def ms(seconds: float) -> float:
    return seconds * 1000.0
