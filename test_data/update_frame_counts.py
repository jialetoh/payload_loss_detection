# running this after organize_videos_ui.py as it assumed constant frame counts

from pathlib import Path
import csv
import cv2


BASE_DIR = Path(__file__).resolve().parent
GROUND_TRUTH_CSV = BASE_DIR / "ground_truth.csv"
VIDEO_DIRS = [BASE_DIR / "videos_loss", BASE_DIR / "videos_normal"]

COLUMNS = ["filename", "camera_id", "is_loss_event", "loss_frame", "total_frames"]


def get_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames


def get_camera_id(video_path: Path) -> str:
    return video_path.name.split("_")[0]


def main():
    if not GROUND_TRUTH_CSV.exists():
        raise FileNotFoundError(f"Missing CSV: {GROUND_TRUTH_CSV}")

    # Build lookup: relative filename -> frame count
    frame_counts = {}

    for video_dir in VIDEO_DIRS:
        if not video_dir.exists():
            continue

        for video_path in video_dir.rglob("*.mp4"):
            rel_path = video_path.relative_to(BASE_DIR).as_posix()
            frame_counts[rel_path] = get_frame_count(video_path)

    updated_rows = []
    missing_files = []

    with GROUND_TRUTH_CSV.open("r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            filename = row["filename"]

            if filename not in frame_counts:
                missing_files.append(filename)
                updated_rows.append(row)
                continue

            row["total_frames"] = frame_counts[filename]
            updated_rows.append(row)

    backup_path = BASE_DIR / "ground_truth_backup.csv"
    GROUND_TRUTH_CSV.replace(backup_path)

    with GROUND_TRUTH_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(updated_rows)

    print(f"Updated frame counts for {len(frame_counts)} videos.")
    print(f"Backup saved as: {backup_path.name}")

    if missing_files:
        print("\nFiles listed in CSV but not found:")
        for file in missing_files:
            print(f"- {file}")


if __name__ == "__main__":
    main()

    