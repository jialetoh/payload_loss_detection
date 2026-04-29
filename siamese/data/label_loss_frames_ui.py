# label_loss_frames_ui.py

from pathlib import Path
import sys

import cv2
import pandas as pd

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QComboBox, QSlider,
    QVBoxLayout, QHBoxLayout, QMessageBox
)


CSV_COLUMNS = ["filename", "camera_id", "is_loss_event", "loss_frame", "total_frames"]


class LossFrameLabeler(QWidget):
    def __init__(self):
        super().__init__()

        self.base_dir = Path(__file__).resolve().parent
        self.video_loss_dir = self.base_dir / "videos_loss"
        self.video_normal_dir = self.base_dir / "videos_normal"
        self.csv_path = self.base_dir / "ground_truth.csv"

        self.video_paths = []
        self.current_video_path = None
        self.cap = None

        self.total_frames = 1
        self.current_frame = 1
        self.is_playing = False

        self.df = None

        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)

        self.setWindowTitle("Loss Frame Labeler")
        self.resize(900, 600)

        self.regenerate_ground_truth_csv()

        self.build_ui()
        self.load_videos()

    def regenerate_ground_truth_csv(self):
        rows = []

        # Label normal videos automatically
        if self.video_normal_dir.exists():
            for video_path in sorted(self.video_normal_dir.glob("*.mp4")):
                rows.append({
                    "filename": video_path.relative_to(self.base_dir).as_posix(),
                    "camera_id": self.get_camera_id(video_path),
                    "is_loss_event": 0,
                    "loss_frame": -1,
                    "total_frames": self.get_total_frames(video_path),
                })

        # Add loss videos with placeholder loss_frame
        if self.video_loss_dir.exists():
            for video_path in sorted(self.video_loss_dir.glob("*.mp4")):
                rows.append({
                    "filename": video_path.relative_to(self.base_dir).as_posix(),
                    "camera_id": self.get_camera_id(video_path),
                    "is_loss_event": 1,
                    "loss_frame": -1,
                    "total_frames": self.get_total_frames(video_path),
                })

        self.df = pd.DataFrame(rows, columns=CSV_COLUMNS)
        self.df.to_csv(self.csv_path, index=False)

    def get_camera_id(self, video_path):
        return video_path.name.split("_")[0]

    def get_total_frames(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total

    def build_ui(self):
        main_layout = QHBoxLayout(self)

        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        self.video_label = QLabel("No video loaded")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        left_layout.addWidget(self.video_label)

        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setMinimum(1)
        self.seek_slider.setValue(1)
        self.seek_slider.sliderMoved.connect(self.seek_to_frame)
        left_layout.addWidget(self.seek_slider)

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

        right_layout.addWidget(QLabel("CSV"))
        self.csv_label = QLabel(str(self.csv_path.name))
        right_layout.addWidget(self.csv_label)

        right_layout.addWidget(QLabel("Loss Video"))
        self.video_combo = QComboBox()
        self.video_combo.currentIndexChanged.connect(self.video_changed)
        right_layout.addWidget(self.video_combo)

        self.current_csv_label = QLabel("Current CSV loss_frame: -1")
        self.selected_loss_label = QLabel("Selected loss_frame: 1")
        self.total_frames_label = QLabel("Total frames: -")

        right_layout.addWidget(self.current_csv_label)
        right_layout.addWidget(self.selected_loss_label)
        right_layout.addWidget(self.total_frames_label)

        self.confirm_btn = QPushButton("Confirm Loss Frame")
        self.confirm_btn.clicked.connect(self.confirm_update)
        right_layout.addWidget(self.confirm_btn)

        right_layout.addStretch()

        main_layout.addLayout(left_layout, stretch=3)
        main_layout.addLayout(right_layout, stretch=1)

    def load_videos(self):
        if not self.video_loss_dir.exists():
            QMessageBox.warning(
                self,
                "Missing Folder",
                f"Folder not found:\n{self.video_loss_dir}",
            )
            return

        self.video_paths = sorted(self.video_loss_dir.glob("*.mp4"))

        self.video_combo.blockSignals(True)
        self.video_combo.clear()

        for path in self.video_paths:
            self.video_combo.addItem(path.name)

        self.video_combo.blockSignals(False)

        if not self.video_paths:
            QMessageBox.information(
                self,
                "No Loss Videos Found",
                "No .mp4 files found in videos_loss/.\n\n"
                "ground_truth.csv was still regenerated for videos_normal/.",
            )
            self.clear_video()
            return

        self.load_video(0)

    def video_changed(self, index):
        if index >= 0:
            self.load_video(index)

    def load_video(self, index):
        self.stop_playback()
        self.release_video()

        self.current_video_path = self.video_paths[index]

        self.cap = cv2.VideoCapture(str(self.current_video_path))
        if not self.cap.isOpened():
            QMessageBox.critical(
                self,
                "Video Error",
                f"Could not open video:\n{self.current_video_path.name}",
            )
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 1

        self.seek_slider.setMaximum(max(1, self.total_frames))
        self.seek_slider.setValue(1)

        self.show_frame(1)
        self.update_csv_info()

    def clear_video(self):
        self.video_label.setText("No video loaded")
        self.video_label.setPixmap(QPixmap())
        self.frame_label.setText("Frame: 1 / 1")
        self.current_csv_label.setText("Current CSV loss_frame: -")
        self.selected_loss_label.setText("Selected loss_frame: -")
        self.total_frames_label.setText("Total frames: -")

    def release_video(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def show_frame(self, frame_number):
        if self.cap is None:
            return

        frame_number = max(1, min(frame_number, self.total_frames))
        self.current_frame = frame_number

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        ok, frame = self.cap.read()

        if not ok:
            self.video_label.setText("Frame read failed")
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape

        qimg = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        self.video_label.setPixmap(pixmap)

        self.seek_slider.blockSignals(True)
        self.seek_slider.setValue(frame_number)
        self.seek_slider.blockSignals(False)

        self.frame_label.setText(f"Frame: {self.current_frame} / {self.total_frames}")
        self.selected_loss_label.setText(f"Selected loss_frame: {self.current_frame}")

    def seek_to_frame(self, value):
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

    def find_matching_row_index(self):
        if self.df is None or self.current_video_path is None:
            return None

        rel_path = self.current_video_path.relative_to(self.base_dir).as_posix()

        matches = self.df[self.df["filename"] == rel_path]

        if len(matches) == 0:
            return None

        return matches.index[0]

    def update_csv_info(self):
        if self.df is None or self.current_video_path is None:
            self.current_csv_label.setText("Current CSV loss_frame: -")
            self.total_frames_label.setText("Total frames: -")
            return

        row_idx = self.find_matching_row_index()

        if row_idx is None:
            self.current_csv_label.setText("Current CSV loss_frame: not found")
            self.total_frames_label.setText(f"Total frames: {self.total_frames}")
            return

        row = self.df.loc[row_idx]

        self.current_csv_label.setText(f"Current CSV loss_frame: {row['loss_frame']}")
        self.total_frames_label.setText(f"Total frames: {self.total_frames}")

    def confirm_update(self):
        if self.df is None:
            QMessageBox.warning(self, "No CSV", "ground_truth.csv was not loaded.")
            return

        if self.current_video_path is None:
            QMessageBox.warning(self, "No Video", "No video is loaded.")
            return

        row_idx = self.find_matching_row_index()

        if row_idx is None:
            QMessageBox.warning(
                self,
                "Row Not Found",
                f"No matching row found in CSV for:\n{self.current_video_path.name}",
            )
            return

        self.df.at[row_idx, "loss_frame"] = int(self.current_frame)
        self.df.at[row_idx, "is_loss_event"] = 1
        self.df.at[row_idx, "total_frames"] = int(self.total_frames)

        self.df.to_csv(self.csv_path, index=False)

        QMessageBox.information(
            self,
            "Updated",
            f"Updated loss_frame for {self.current_video_path.name} to {self.current_frame}.",
        )

        self.mark_current_done_and_next()

    def mark_current_done_and_next(self):
        current_index = self.video_combo.currentIndex()

        current_text = self.video_combo.itemText(current_index)
        if not current_text.endswith(" ✓"):
            self.video_combo.setItemText(current_index, current_text + " ✓")

        next_index = current_index + 1

        if next_index < self.video_combo.count():
            self.video_combo.setCurrentIndex(next_index)
        else:
            QMessageBox.information(self, "Done", "Reached the last loss video.")

    def closeEvent(self, event):
        self.stop_playback()
        self.release_video()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = LossFrameLabeler()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()