import sys
import os
import logging
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QTextEdit, QProgressBar,
    QFileDialog, QMessageBox, QGroupBox, QGridLayout
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont

import numpy as np
from PySide6.QtMultimedia import QMediaDevices, QAudioSource, QAudioFormat


# load style sheet
def load_styles(app):
    """Load external Qt Style Sheet (QSS)."""
    qss_path = os.path.join(os.path.dirname(__file__), "style.qss")
    if os.path.exists(qss_path):
        with open(qss_path, "r") as f:
            app.setStyleSheet(f.read())
            print(f"0 - Loaded QSS: {qss_path}")
    else:
        print("x - style.qss not found. Running without stylesheet.")


# Main Application Window
class MainWindow(QMainWindow):
    """Main window for the Whisper Transcription App."""

    def __init__(self):
        super().__init__()
        self.audio_capture = None
        self.audio_buffer = None
        self.transcription_thread = None
        self.is_transcribing = False
        self.transcript_text = ""

        self.setup_ui()
        self.setup_audio()
        self.setup_transcription()
        self.update_ui_state()


    def setup_ui(self):
        self.setWindowTitle(" ")
        self.setGeometry(100, 100, 900, 700)

        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main Layout
        main_layout = QVBoxLayout(central_widget)


        self.title = QLabel("Whisper Real-time Diarization & Transcription App")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title.setObjectName("mainTitle")
        main_layout.addWidget(self.title)


        # Setting Group
        setting_group = QGroupBox("Settings")
        setting_layout = QGridLayout(setting_group)
        # make left and right halves balanced
        setting_layout.setColumnStretch(0, 1)  # label column (left)
        setting_layout.setColumnStretch(1, 2)  # combo box  (left)
        setting_layout.setColumnStretch(2, 1)  # label column (right)
        setting_layout.setColumnStretch(3, 2)  # combo box (right)


        # row 0
        # Audio Device
        setting_layout.addWidget(QLabel("Audio Device:"), 0, 0)
        self.device_combo = QComboBox()
        self.device_combo.currentTextChanged.connect(self.on_device_changed)
        setting_layout.addWidget(self.device_combo, 0, 1)

        # Volume Bar
        setting_layout.addWidget(QLabel("Microphone Level:"), 0, 2)
        self.volume_bar = QProgressBar()
        self.volume_bar.setRange(0, 100)
        self.volume_bar.setValue(0)

        setting_layout.addWidget(self.volume_bar, 0, 3)

        # row 1
        # Real-time Model
        setting_layout.addWidget(QLabel("Real-Time Model:"), 1, 0)
        self.rt_model_combo = QComboBox()
        self.rt_model_combo.currentTextChanged.connect(self.on_rt_model_changed)
        setting_layout.addWidget(self.rt_model_combo, 1, 1)

        self.rt_model_info_label = QLabel("")
        self.rt_model_info_label.setWordWrap(True)
        setting_layout.addWidget(self.rt_model_info_label, 2, 1)

        # Diarization Model
        setting_layout.addWidget(QLabel("Diarization Model:"), 1, 2)
        self.dz_model_combo = QComboBox()
        self.dz_model_combo.currentTextChanged.connect(self.on_dz_model_changed)
        setting_layout.addWidget(self.dz_model_combo, 1, 3)

        self.dz_model_info_label = QLabel("")
        self.dz_model_info_label.setWordWrap(True)
        setting_layout.addWidget(self.dz_model_info_label, 2, 3)

        main_layout.addWidget(setting_group)

        # Transcription Output
        output_group = QGroupBox("Transcription Output")
        output_layout = QVBoxLayout(output_group)

        self.transcript_display = QTextEdit()
        self.transcript_display.setFont(QFont("Consolas", 10))
        self.transcript_display.setPlaceholderText("Transcription will appear here...")
        output_layout.addWidget(self.transcript_display)

        main_layout.addWidget(output_group)

        # Control Buttons
        control_layout = QHBoxLayout()

        self.start_button = QPushButton("▶ Start")
        self.start_button.setObjectName("startButton")
        self.start_button.clicked.connect(self.start_transcription)
        control_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("■ Stop")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.clicked.connect(self.stop_transcription)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.stop_button)

        self.save_button = QPushButton("💾 Save")
        self.save_button.setObjectName("saveButton")
        self.save_button.clicked.connect(self.save_transcript)
        control_layout.addWidget(self.save_button)

        main_layout.addLayout(control_layout)

        # Status Bar
        self.statusBar().showMessage("Ready")

    # AUDIO SETUP
    def setup_audio(self):
        """Set up audio device list and connect change signals."""
        self.media_devices = QMediaDevices()
        self.media_devices.audioInputsChanged.connect(self.populate_devices)
        self.populate_devices()
        self.start_audio_monitoring()


    def populate_devices(self):
        """Detect and populate available audio input devices."""
        self.device_combo.blockSignals(True)  # prevent triggering change while filling
        self.device_combo.clear()

        devices = QMediaDevices.audioInputs()

        if not devices:
            self.device_combo.addItem("No audio input found")
            self.device_combo.setEnabled(False)
            self.statusBar().showMessage("No microphones detected.")
        else:
            for dev in devices:
                self.device_combo.addItem(dev.description())
            self.device_combo.setEnabled(True)

            # Default device
            default_device = QMediaDevices.defaultAudioInput()
            default_name = default_device.description()
            index = self.device_combo.findText(default_name)
            if index >= 0:
                self.device_combo.setCurrentIndex(index)
                self.statusBar().showMessage(f"Default mic: {default_name}")
            else:
                self.statusBar().showMessage("Select a microphone to begin.")

        self.device_combo.blockSignals(False)

    def on_device_changed(self): 
        selected_device_name = self.device_combo.currentText()
        devices = QMediaDevices.audioInputs()
        for dev in devices:
            if dev.description() == selected_device_name:
                self.audio_capture = dev
                self.statusBar().showMessage(f"Selected mic: {selected_device_name}")
                self.start_audio_monitoring(dev)
                break

    def start_audio_monitoring(self, device=None):
        # Stop old stream if running
        if hasattr(self, "audio_source") and self.audio_source:
            self.audio_source.stop()

        # Choose selected or default device
        if device is None:
            selected_name = self.device_combo.currentText()
            devices = QMediaDevices.audioInputs()
            device = next((d for d in devices if d.description() == selected_name),
                        QMediaDevices.defaultAudioInput())

        fmt = QAudioFormat()
        fmt.setSampleRate(16000) # sample_rate
        fmt.setChannelCount(1)
        fmt.setSampleFormat(QAudioFormat.Int16)

    
        self.audio_source = QAudioSource(device, fmt)
        self.io_device = self.audio_source.start()

        # timer for update vlome bar
        if hasattr(self, "timer"):
            self.timer.stop()
        self.timer = QTimer()
        self.timer.timeout.connect(self.read_audio_data)
        self.timer.start(100)  # update every 100ms

        self.statusBar().showMessage(f"Monitoring: {device.description()}")


    def read_audio_data(self):
        if not hasattr(self, "io_device") or not self.io_device:
            return

        data = self.io_device.readAll()
        if data.size() == 0:
            self.volume_bar.setValue(0)
            return

        samples = np.frombuffer(data.data(), dtype=np.int16).astype(np.float32)
        if len(samples) == 0:
            return

        # Calculate RMS and update volume bar
        # rms = np.sqrt(np.mean(samples**2)) root mean square
        rms = np.sqrt(np.mean(samples ** 2))
        level = min(int((rms / 32768.0) * 100), 100) # 16khz int16 max

        self.volume_bar.setValue(level)


# ============================
# Main Entry
# ============================
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Whisper Voice to Text")

    load_styles(app)  # ✅ Load QSS stylesheet

    window = MainWindow()
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()