from pathlib import Path
import random

import cv2
import numpy as np
import pandas as pd
import torch


# -------------------------
# Paths
# -------------------------

SIAMESE_DIR = Path(__file__).resolve().parent
DATA_DIR = SIAMESE_DIR / "data"

VIDEOS_LOSS_DIR = DATA_DIR / "videos_loss"
VIDEOS_NORMAL_DIR = DATA_DIR / "videos_normal"
EXTRACTED_PAIRS_DIR = DATA_DIR / "extracted_pairs"

GROUND_TRUTH_CSV = DATA_DIR / "ground_truth.csv"
TRAIN_LABELS_CSV = EXTRACTED_PAIRS_DIR / "train_labels.csv"

WEIGHTS_DIR = SIAMESE_DIR.parent / "weights"
SIAMESE_WEIGHTS_PATH = WEIGHTS_DIR / "siamese_best.pth"


# -------------------------
# Parameters
# -------------------------

IGNORE_FIRST_N_FRAMES = 30
BUFFER_FRAMES = 35
FRAME_STEP = 5


# -------------------------
# General helpers
# -------------------------

def set_seed(seed: int = 42):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device():
    """Return best available device: CUDA, MPS, or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def ensure_dirs():
    """Create required output directories."""
    EXTRACTED_PAIRS_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


def read_ground_truth() -> pd.DataFrame:
    """Load Siamese training ground truth CSV."""
    return pd.read_csv(GROUND_TRUTH_CSV)


def read_train_labels() -> pd.DataFrame:
    """Load extracted Siamese pair labels."""
    return pd.read_csv(TRAIN_LABELS_CSV)


def get_dl_kwargs():
    """Return DataLoader kwargs based on device."""
    device = get_device()

    if device.type == "cuda":
        return {"num_workers": 4, "pin_memory": True}

    return {"num_workers": 0, "pin_memory": False}


# -------------------------
# Video helpers
# -------------------------

def get_video_path(filename: str) -> Path:
    """
    Resolve video path from ground_truth.csv.

    Expected filename examples:
    - videos_loss/front_toolbox_loss_toolbox_001.mp4
    - videos_normal/front_toolbox_001.mp4
    """
    return DATA_DIR / filename


def open_video(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    return cap


def iter_sampled_video_frames(
    video_path: Path,
    ignore_first_n_frames: int = IGNORE_FIRST_N_FRAMES,
    frame_step: int = FRAME_STEP,
):
    """
    Yield sampled frames as (frame_idx, frame).

    frame_idx starts from 1.
    """
    cap = open_video(video_path)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        if frame_idx <= ignore_first_n_frames:
            continue

        if frame_idx % frame_step != 0:
            continue

        yield frame_idx, frame

    cap.release()


def split_loss_frames(
    frames,
    loss_frame: int,
    buffer_frames: int = BUFFER_FRAMES,
):
    """
    Split sampled frames into safe before/after regions.

    Frames within ±buffer_frames of loss_frame are excluded.
    """
    before = []
    after = []

    for frame_idx, frame in frames:
        if frame_idx < loss_frame - buffer_frames:
            before.append((frame_idx, frame))
        elif frame_idx > loss_frame + buffer_frames:
            after.append((frame_idx, frame))

    return before, after


# -------------------------
# Pair helpers
# -------------------------

def make_random_pairs(pool_a, pool_b, label: int, num_pairs: int):
    """
    Randomly create frame pairs from two frame pools.

    Each pool contains tuples of (frame_idx, frame).
    Returns tuples of (idx1, frame1, idx2, frame2, label).
    """
    if len(pool_a) == 0 or len(pool_b) == 0:
        return []

    pairs = []

    for _ in range(num_pairs):
        idx1, frame1 = random.choice(pool_a)
        idx2, frame2 = random.choice(pool_b)

        if pool_a is pool_b and len(pool_a) > 1:
            attempts = 0
            while idx1 == idx2 and attempts < 10:
                idx2, frame2 = random.choice(pool_b)
                attempts += 1

        pairs.append((idx1, frame1, idx2, frame2, label))

    return pairs


def save_pair_images(frame1, frame2, pair_id: int):
    """
    Save one image pair into extracted_pairs/.

    Returns relative paths suitable for train_labels.csv.
    """
    img1_path = EXTRACTED_PAIRS_DIR / f"pair_{pair_id:06d}_1.jpg"
    img2_path = EXTRACTED_PAIRS_DIR / f"pair_{pair_id:06d}_2.jpg"

    cv2.imwrite(str(img1_path), frame1)
    cv2.imwrite(str(img2_path), frame2)

    return (
        img1_path.relative_to(SIAMESE_DIR).as_posix(),
        img2_path.relative_to(SIAMESE_DIR).as_posix(),
    )


def clear_extracted_pairs():
    """Delete old extracted pair images and train_labels.csv."""
    if not EXTRACTED_PAIRS_DIR.exists():
        EXTRACTED_PAIRS_DIR.mkdir(parents=True, exist_ok=True)
        return

    for path in EXTRACTED_PAIRS_DIR.glob("*.jpg"):
        path.unlink()

    if TRAIN_LABELS_CSV.exists():
        TRAIN_LABELS_CSV.unlink()


# -------------------------
# Image helpers
# -------------------------

def read_image_bgr(image_path: Path):
    img = cv2.imread(str(image_path))

    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    return img


def get_image_info(image_path: Path) -> dict:
    img = read_image_bgr(image_path)

    return {
        "shape": img.shape,
        "dtype": img.dtype,
    }

