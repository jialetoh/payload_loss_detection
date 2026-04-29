from pathlib import Path
import csv
import re
import shutil
import sys

import cv2
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QComboBox, QRadioButton,
    QButtonGroup, QSlider, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLineEdit, QMessageBox, QGroupBox, QFileDialog
)


CAMERAS = ["front", "back", "left", "right"]
LIGHTING_MAP = {
    "Low": "indoor_low",
    "Medium": "indoor_med",
    "Full": "indoor_full",
}
CSV_COLUMNS = ["filename", "camera_id", "is_loss_event", "loss_frame", "total_frames"]


class VideoGroup:
    def __init__(self, group_id: str, files: dict[str, Path]):
        self.group_id = group_id
        self.files = files


class VideoSorter(QWidget):
    def __init__(self):
        super().__init__()

        self.base_dir = Path(__file__).resolve().parent
        self.groups: list[VideoGroup] = []
        self.current_group: VideoGroup | None = None

        self.captures = {}
        self.total_frames = 0
        self.current_frame = 1
        self.is_playing = False

        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)

        self.setWindowTitle("Payload Video Sorter")
        self.resize(1000, 650)

        self.build_ui()
        self.ensure_folders()
        self.load_csv_files()
        self.load_video_groups()

    def build_ui(self):
        main_layout = QHBoxLayout(self)

        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        # Video display
        self.video_labels = {}
        grid = QGridLayout()

        positions = {
            "front": (0, 0),
            "back": (0, 1),
            "left": (1, 0),
            "right": (1, 1),
        }

        for cam, pos in positions.items():
            label = QLabel(f"{cam}")
            label.setAlignment(Qt.AlignCenter)
            label.setFixedSize(320, 240)
            label.setStyleSheet("background-color: black; color: white;")
            self.video_labels[cam] = label
            grid.addWidget(label, *pos)

        left_layout.addLayout(grid)

        # Seek bar
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setMinimum(1)
        self.seek_slider.setValue(1)
        self.seek_slider.sliderMoved.connect(self.seek_to_frame)
        left_layout.addWidget(self.seek_slider)

        # Controls
        controls = QHBoxLayout()

        self.prev_btn = QPushButton("|<")
        self.play_btn = QPushButton("▶︎")
        self.next_btn = QPushButton(">|")
        self.frame_label = QLabel("Frame: 1 / 1")

        self.prev_btn.clicked.connect(self.prev_frame)
        self.play_btn.clicked.connect(self.toggle_play)
        self.next_btn.clicked.connect(self.next_frame)

        controls.addWidget(self.prev_btn)
        controls.addWidget(self.play_btn)
        controls.addWidget(self.next_btn)
        controls.addWidget(self.frame_label)
        controls.addStretch()

        left_layout.addLayout(controls)

        # Right controls
        self.csv_combo = QComboBox()
        self.csv_combo.currentTextChanged.connect(self.csv_changed)

        self.group_combo = QComboBox()
        self.group_combo.currentIndexChanged.connect(self.group_changed)

        right_layout.addWidget(QLabel("Ground Truth CSV"))
        right_layout.addWidget(self.csv_combo)

        right_layout.addWidget(QLabel("Video Group"))
        right_layout.addWidget(self.group_combo)

        # Event class
        event_box = QGroupBox("Event Class")
        event_layout = QVBoxLayout()

        self.loss_radio = QRadioButton("Payload Loss")
        self.normal_radio = QRadioButton("Normal Operation")
        self.normal_radio.setChecked(True)

        self.event_group = QButtonGroup()
        self.event_group.addButton(self.loss_radio)
        self.event_group.addButton(self.normal_radio)

        self.loss_radio.toggled.connect(self.update_loss_controls)

        event_layout.addWidget(self.normal_radio)
        event_layout.addWidget(self.loss_radio)
        event_box.setLayout(event_layout)
        right_layout.addWidget(event_box)

        # Lighting
        lighting_box = QGroupBox("Lighting Condition")
        lighting_layout = QHBoxLayout()

        self.low_radio = QRadioButton("Low")
        self.med_radio = QRadioButton("Medium")
        self.full_radio = QRadioButton("Full")
        self.low_radio.setChecked(True)

        self.lighting_group = QButtonGroup()
        self.lighting_group.addButton(self.low_radio)
        self.lighting_group.addButton(self.med_radio)
        self.lighting_group.addButton(self.full_radio)

        lighting_layout.addWidget(self.low_radio)
        lighting_layout.addWidget(self.med_radio)
        lighting_layout.addWidget(self.full_radio)

        lighting_box.setLayout(lighting_layout)
        right_layout.addWidget(lighting_box)

        # Scenario / lost item
        right_layout.addWidget(QLabel("Scenario"))
        self.scenario_input = QLineEdit()
        self.scenario_input.setPlaceholderText("e.g. empty, bottle1, toolbox")
        right_layout.addWidget(self.scenario_input)

        self.lostitem_label = QLabel("Lost Item")
        right_layout.addWidget(self.lostitem_label)

        self.lostitem_input = QLineEdit()
        self.lostitem_input.setPlaceholderText("e.g. bottle1, toolbox, mtape")
        right_layout.addWidget(self.lostitem_input)

        self.lostitem_label.hide()
        self.lostitem_input.hide()

        self.loss_frame_label = QLabel("Loss frame: -1")
        right_layout.addWidget(self.loss_frame_label)

        self.confirm_btn = QPushButton("Confirm and Sort")
        self.confirm_btn.clicked.connect(self.confirm_sort)
        right_layout.addWidget(self.confirm_btn)

        right_layout.addStretch()

        main_layout.addLayout(left_layout, stretch=3)
        main_layout.addLayout(right_layout, stretch=1)

    def ensure_folders(self):
        for event_folder in ["videos_loss", "videos_normal"]:
            for lighting in LIGHTING_MAP.values():
                (self.base_dir / event_folder / lighting).mkdir(parents=True, exist_ok=True)

    def load_csv_files(self):
        self.csv_combo.blockSignals(True)
        self.csv_combo.clear()

        csv_files = sorted(self.base_dir.glob("*.csv"))

        if not csv_files:
            default_csv = self.base_dir / "ground_truth.csv"
            self.create_csv_if_missing(default_csv)
            csv_files = [default_csv]

        names = [p.name for p in csv_files]
        self.csv_combo.addItems(names)

        if "ground_truth.csv" in names:
            self.csv_combo.setCurrentText("ground_truth.csv")

        self.csv_combo.blockSignals(False)

    def csv_changed(self):
        csv_path = self.base_dir / self.csv_combo.currentText()
        self.create_csv_if_missing(csv_path)

    def create_csv_if_missing(self, csv_path: Path):
        if not csv_path.exists():
            with csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()

    def load_video_groups(self):
        pattern = re.compile(r"^(front|back|left|right)_(\d+)\.mp4$", re.IGNORECASE)
        found = {}

        for video in sorted(self.base_dir.glob("*.mp4")):
            match = pattern.match(video.name)
            if not match:
                continue

            cam, group_id = match.groups()
            cam = cam.lower()
            found.setdefault(group_id, {})[cam] = video

        self.groups = []
        missing_messages = []

        for group_id, files in sorted(found.items()):
            missing = [cam for cam in CAMERAS if cam not in files]
            if missing:
                missing_messages.append(f"Group {group_id}: missing {', '.join(missing)}")
                continue

            self.groups.append(VideoGroup(group_id, files))

        self.group_combo.blockSignals(True)
        self.group_combo.clear()

        for group in self.groups:
            self.group_combo.addItem(group.group_id)

        self.group_combo.blockSignals(False)

        if missing_messages:
            QMessageBox.warning(
                self,
                "Incomplete Video Groups",
                "Some groups were skipped because camera views are missing:\n\n"
                + "\n".join(missing_messages),
            )

        if not self.groups:
            self.clear_video_display()
            QMessageBox.information(
                self,
                "No Videos Found",
                "No complete video groups were found in this folder.\n\n"
                "Expected format: front_001.mp4, back_001.mp4, left_001.mp4, right_001.mp4",
            )
            return

        self.load_group(0)

    def group_changed(self, index):
        if index >= 0:
            self.load_group(index)

    def load_group(self, index: int):
        self.stop_playback()
        self.release_captures()

        self.current_group = self.groups[index]
        self.captures = {}

        total_frames_list = []

        for cam, path in self.current_group.files.items():
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                QMessageBox.critical(self, "Video Error", f"Could not open {path.name}")
                return

            self.captures[cam] = cap
            total_frames_list.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

        self.total_frames = min(total_frames_list) if total_frames_list else 1
        self.current_frame = 1

        self.seek_slider.setMaximum(max(1, self.total_frames))
        self.seek_slider.setValue(1)

        self.show_frame(1)

    def clear_video_display(self):
        for cam, label in self.video_labels.items():
            label.setText(cam)
            label.setPixmap(QPixmap())

    def release_captures(self):
        for cap in self.captures.values():
            cap.release()
        self.captures = {}

    def show_frame(self, frame_number: int):
        if not self.captures:
            return

        frame_number = max(1, min(frame_number, self.total_frames))
        self.current_frame = frame_number

        for cam, cap in self.captures.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
            ok, frame = cap.read()

            if not ok:
                self.video_labels[cam].setText(f"{cam}\nFrame read failed")
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape

            qimg = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg).scaled(
                self.video_labels[cam].size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )

            self.video_labels[cam].setPixmap(pixmap)

        self.seek_slider.blockSignals(True)
        self.seek_slider.setValue(frame_number)
        self.seek_slider.blockSignals(False)

        self.frame_label.setText(f"Frame: {self.current_frame} / {self.total_frames}")
        self.update_loss_frame_label()

    def seek_to_frame(self, value: int):
        self.show_frame(value)

    def prev_frame(self):
        self.stop_playback()
        self.show_frame(self.current_frame - 1)

    def next_frame(self):
        if self.current_frame >= self.total_frames:
            self.stop_playback()
            return

        self.show_frame(self.current_frame + 1)

    def toggle_play(self):
        if self.is_playing:
            self.stop_playback()
        else:
            self.is_playing = True
            self.play_btn.setText("⏸")
            self.timer.start(33)

    def stop_playback(self):
        self.is_playing = False
        self.play_btn.setText("▶︎")
        self.timer.stop()

    def update_loss_controls(self):
        is_loss = self.loss_radio.isChecked()

        self.lostitem_label.setVisible(is_loss)
        self.lostitem_input.setVisible(is_loss)

        if not is_loss:
            self.lostitem_input.clear()

        self.update_loss_frame_label()

    def update_loss_frame_label(self):
        if self.loss_radio.isChecked():
            self.loss_frame_label.setText(f"Loss frame: {self.current_frame}")
        else:
            self.loss_frame_label.setText("Loss frame: -1")

    def get_next_trial(self, target_dir: Path, event_class: str, scenario: str, lostitem: str | None):
        if event_class == "loss":
            regex = re.compile(
                rf"^(front|back|left|right)_{re.escape(scenario)}_loss_{re.escape(lostitem)}_(\d+)\.mp4$"
            )
        else:
            regex = re.compile(
                rf"^(front|back|left|right)_{re.escape(scenario)}_(\d+)\.mp4$"
            )

        max_trial = 0

        for file in target_dir.glob("*.mp4"):
            match = regex.match(file.name)
            if match:
                trial = int(match.groups()[-1])
                max_trial = max(max_trial, trial)

        return f"{max_trial + 1:03d}"

    def confirm_sort(self):
        if not self.current_group:
            QMessageBox.warning(self, "No Video Group", "No video group is currently loaded.")
            return

        scenario = self.scenario_input.text().strip().lower().replace(" ", "")
        lostitem = self.lostitem_input.text().strip().lower().replace(" ", "")

        if not scenario:
            QMessageBox.warning(self, "Missing Scenario", "Please enter a scenario.")
            return

        is_loss = self.loss_radio.isChecked()

        if is_loss and not lostitem:
            QMessageBox.warning(self, "Missing Lost Item", "Please enter the lost item.")
            return

        if self.low_radio.isChecked():
            lighting_folder = LIGHTING_MAP["Low"]
        elif self.med_radio.isChecked():
            lighting_folder = LIGHTING_MAP["Medium"]
        else:
            lighting_folder = LIGHTING_MAP["Full"]
        event_folder = "videos_loss" if is_loss else "videos_normal"
        event_class = "loss" if is_loss else "normal"

        target_dir = self.base_dir / event_folder / lighting_folder
        target_dir.mkdir(parents=True, exist_ok=True)

        trial = self.get_next_trial(
            target_dir=target_dir,
            event_class=event_class,
            scenario=scenario,
            lostitem=lostitem if is_loss else None,
        )

        planned_moves = {}

        for cam in CAMERAS:
            old_path = self.current_group.files[cam]

            if is_loss:
                new_name = f"{cam}_{scenario}_loss_{lostitem}_{trial}.mp4"
            else:
                new_name = f"{cam}_{scenario}_{trial}.mp4"

            new_path = target_dir / new_name

            if new_path.exists():
                QMessageBox.critical(
                    self,
                    "File Exists",
                    f"Target file already exists:\n{new_path}",
                )
                return

            planned_moves[cam] = (old_path, new_path)

        loss_frame = self.current_frame if is_loss else -1
        is_loss_event = 1 if is_loss else 0

        confirm_text = (
            f"Sort group {self.current_group.group_id}?\n\n"
            f"Event: {'Payload Loss' if is_loss else 'Normal Operation'}\n"
            f"Lighting: {lighting_folder}\n"
            f"Scenario: {scenario}\n"
            f"Trial: {trial}\n"
            f"Loss frame: {loss_frame}\n\n"
            f"Videos will be moved to:\n{target_dir}"
        )

        reply = QMessageBox.question(
            self,
            "Confirm Sort",
            confirm_text,
            QMessageBox.Yes | QMessageBox.No,
        )

        if reply != QMessageBox.Yes:
            return

        self.stop_playback()
        self.release_captures()

        rows = []

        try:
            for cam, (old_path, new_path) in planned_moves.items():
                shutil.move(str(old_path), str(new_path))

                relative_filename = new_path.relative_to(self.base_dir).as_posix()

                rows.append({
                    "filename": relative_filename,
                    "camera_id": cam,
                    "is_loss_event": is_loss_event,
                    "loss_frame": loss_frame,
                    "total_frames": self.total_frames,
                })

            csv_path = self.base_dir / self.csv_combo.currentText()
            self.append_csv_rows(csv_path, rows)

        except Exception as e:
            QMessageBox.critical(self, "Sort Failed", f"An error occurred:\n{e}")
            return

        QMessageBox.information(self, "Done", "Videos sorted and CSV updated.")

        self.scenario_input.clear()
        self.lostitem_input.clear()

        self.load_csv_files()
        self.load_video_groups()

    def append_csv_rows(self, csv_path: Path, rows: list[dict]):
        self.create_csv_if_missing(csv_path)

        with csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            for row in rows:
                writer.writerow(row)

    def closeEvent(self, event):
        self.stop_playback()
        self.release_captures()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = VideoSorter()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
