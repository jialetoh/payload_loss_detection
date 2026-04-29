import cv2
import os
import time
import threading
import subprocess
from pathlib import Path

# =========================
# CONFIG
# =========================
CAMERA_MAP = {
    0: "right",
    2: "front",
    4: "back",
    6: "left",
}

FRAME_WIDTH = 256
FRAME_HEIGHT = 144
FPS = 10.0

# Local save folder on bot
LOCAL_SAVE_DIR = "/home/isera2/Desktop/payload_loss_recordings"

# Your computer details (destination)
REMOTE_USER = "tohjiale"
REMOTE_HOST = "10.32.18.103"
REMOTE_DIR = "'/Users/tohjiale/Documents/SUTD/18 Term 8/50035 Computer Vision/Project/payload_loss_detection/data/payload_recordings'"

# Video codec
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")

# =========================
# GLOBALS
# =========================
stop_event = threading.Event()


def get_next_counter(save_dir: str) -> int:
    """
    Find the next available recording counter by scanning existing mp4 files.
    Expects filenames like left_001.mp4, front_002.mp4, etc.
    """
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)

    max_counter = 0
    for file in path.glob("*.mp4"):
        stem = file.stem  # e.g. "left_001"
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        try:
            counter = int(parts[1])
            max_counter = max(max_counter, counter)
        except ValueError:
            continue

    return max_counter + 1


def record_camera(cam_idx: int, output_path: str):
    cap = cv2.VideoCapture(cam_idx, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {cam_idx}")
        return
        
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        output_path,
        FOURCC,
        FPS,
        (actual_width, actual_height)
    )

    if not writer.isOpened():
        print(f"[ERROR] Could not open VideoWriter for camera {cam_idx}")
        cap.release()
        return

    print(f"[INFO] Camera {cam_idx} recording to {output_path}")

    frame_interval = 1.0 / FPS
    next_frame_time = time.time()

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print(f"[WARNING] Camera {cam_idx}: failed to read frame")
            time.sleep(0.01)
            continue

        writer.write(frame)

        next_frame_time += frame_interval
        sleep_time = next_frame_time - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)

    writer.release()
    cap.release()
    print(f"[INFO] Camera {cam_idx} stopped.")


def rsync_to_remote(local_dir: str):
    remote_target = f"{REMOTE_USER}@{REMOTE_HOST}:{REMOTE_DIR}/"
    cmd = [
        "rsync",
        "-avzP",
        local_dir + "/",
        remote_target
    ]

    print("[INFO] Sending videos to computer...")
    try:
        subprocess.run(cmd, check=True)
        print("[INFO] Transfer complete.")
    except subprocess.CalledProcessError as e:
        print("[ERROR] rsync failed.")
        print(e)


def run_recording_session(counter: int):
    stop_event.clear()

    output_files = []
    for cam_idx, cam_name in CAMERA_MAP.items():
        filename = f"{cam_name}_{counter:03d}.mp4"
        output_files.append((cam_idx, os.path.join(LOCAL_SAVE_DIR, filename)))

    input(f"\nPress Enter to START recording session {counter:03d}...")

    threads = []
    for cam_idx, output_path in output_files:
        t = threading.Thread(target=record_camera, args=(cam_idx, output_path))
        t.start()
        threads.append(t)

    input("Recording... Press Enter to STOP recording...")

    stop_event.set()

    for t in threads:
        t.join()

    rsync_to_remote(LOCAL_SAVE_DIR)


def main():
    os.makedirs(LOCAL_SAVE_DIR, exist_ok=True)

    counter = get_next_counter(LOCAL_SAVE_DIR)
    print(f"[INFO] Starting from recording counter {counter:03d}")

    while True:
        run_recording_session(counter)
        counter += 1

        choice = input("\nPress Enter to record again, or type 'q' then Enter to quit: ").strip().lower()
        if choice == "q":
            print("[INFO] Exiting.")
            break


if __name__ == "__main__":
    main()

