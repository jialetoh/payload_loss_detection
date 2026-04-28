# Payload Loss Detection using Computer Vision

This project implements and compares different computer vision approaches for detecting payload loss on an autonomous logistics robot. The system monitors the payload shelf using four camera views and outputs a binary payload state: **Payload Present** or **Payload Absent**.


## Project Overview

The bot transports tools, toolboxes, equipment, and supplies within an industrial environment, during which payloads may shift, fall off, or be removed. This project explores a computer vision-based payload monitoring system to detect such events in real time.

Three approaches were evaluated:

1. **Structural Similarity Index (SSIM)**
2. **YOLO26n Object Detection**
3. **Custom Siamese Network** (final chosen method)

The final comparison was conducted on a test dataset of 144 videos.


## Folder Structure

```text
payload_loss_detection/
├── README.md
├── requirements.txt
│
├── test_data/
│   ├── videos_loss/
│   │   ├── indoor_low/
│   │   ├── indoor_med/
│   │   └── indoor_full/
│   ├── videos_normal/
│   │   ├── indoor_low/
│   │   ├── indoor_med/
│   │   └── indoor_full/
│   └── ground_truth.csv
│
├── siamese/
│   ├── data/
│   │   ├── videos_loss/
│   │   ├── videos_normal/
│   │   ├── extracted_pairs/
│   │   └── train_labels.csv
│   ├── 01_prepare_pairs.ipynb
│   ├── 02_train_siamese.ipynb
│   └── siamese_model.py
│
├── src/
│   ├── base_method.py
│   ├── ssim_method.py
│   ├── yolo_method.py
│   ├── siamese_method.py
│   ├── eval.py
│   ├── metrics.py
│   └── utils.py
│
├── notebooks/
│   ├── 01_test_ssim.ipynb
│   ├── 02_test_yolo.ipynb
│   ├── 03_test_siamese.ipynb
│   └── 04_compare_all_methods.ipynb
│
├── weights/
│   ├── siamese_best.pth
│   └── yolo26n.pt
│
└── deployment/
```


## Test Dataset

The test dataset contains 144 videos, each approximately 30 to 40 seconds long, generated from:
```text
3 lighting conditions × 6 payload scenarios × 2 event classes × 4 camera views
```

The ground truth file is stored at: `test_data/ground_truth.csv`.

It contains the following fields:

| Field | Description |
|---|---|
| `filename` | Name of the video file |
| `camera_id` | Camera view: front, back, left, or right |
| `is_loss_event` | `0` for normal operation, `1` for payload loss |
| `loss_frame` | Frame where payload loss occurs, or `-1` if no loss occurs |
| `total_frames` | Total number of frames in the video |


## Siamese Network

The Siamese network uses a **separate training dataset** from the shared test dataset.

Training data includes:

- 104 payload loss videos
- 32 normal operation videos

Normal-operation frame pairs can also be sampled from payload loss videos, as long as both frames are taken either before or after the loss event (no need to record so many normal-operation videos).

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Then use the notebooks in the `notebooks/` folder to test and compare the different approaches. The notebooks under `siamese/` are used only to prepare data for and train the Siamese network.

## Notes

- The shared `test_data/` folder is used only for final testing and comparison.
- The `siamese/` folder contains separate data and notebooks for training the Siamese network.
- The SSIM and YOLO26n methods do not require training.
- Video frames should be processed in memory and not stored permanently during deployment.
- The `deployment/` folder contains files that can be run directly on the bot using `./run_demo.sh`.
