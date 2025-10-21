from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QPushButton, QTextEdit, QProgressBar, QGroupBox,
    QGridLayout, QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QTextCursor, QTextCharFormat, QFont, QColor, QShortcut, QKeySequence
from PySide6.QtMultimedia import QMediaDevices, QAudioSource, QAudioFormat

import threading
import yaml
import numpy as np
import os

from backend.QtStreamer import QAudioStreamer as ms
from backend import AsrWorker as aw
from backend import VadUtils as vadu
from backend import AsrModel as am
from backend import DiarizationUtil as du


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(" ")
        self.setMinimumSize(900, 620)

        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

        # === BACKEND ===
        self.mic = ms(config["mic"])
        self.asr_model = am.NvidiaParakeet(config["asr"])
        self.vad = vadu.VadUtils(config["vad"])
        self.diarize = du.DiarizationUtil(config["diarize"])
        self.asr_worker = aw.ParakeetAsrWorker(
            self.mic, self.asr_model, self.vad, self.diarize, config["asr_worker"]
        )

        # === Keyboard Shortcuts ===
        QShortcut(QKeySequence("Space"), self, activated=self.toggleTranscription)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self.saveText)      
        QShortcut(QKeySequence("Ctrl+Q"), self, activated=self.close)     
        QShortcut(QKeySequence("Ctrl+X"), self, activated=self.clearTranscript)

        # === FRONTEND AUDIO ===
        self.audio_source = None
        self.io_device = None
        self.timer = None

        # === UI ===
        self._partial_block_num = None
        self.setup_ui()

        # Device list
        self.media_devices = QMediaDevices()
        self.media_devices.audioInputsChanged.connect(self.populate_devices)
        self.populate_devices()

        self.statusBar().showMessage("Ready")
        self.asr_t = None

    # UI ===============================================================
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        title = QLabel("Whisper Real-time Diarization & Transcription App")
        title.setAlignment(Qt.AlignCenter)
        title.setObjectName("mainTitle")
        main_layout.addWidget(title)

        # === UI-Settings ===
        setting_group = QGroupBox("Microphone Settings")
        setting_layout = QGridLayout(setting_group)
        setting_layout.setContentsMargins(10, 5, 10, 5)
        setting_layout.setSpacing(8)

        # --- UI-Settings-Left (Audio Device) ---
        left_widget = QWidget()
        left_layout = QHBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        lbl_device = QLabel("Audio Device:")
        lbl_device.setObjectName("deviceLabel")

        self.device_combo = QComboBox()
        self.device_combo.setObjectName("audioDeviceCombo")
        self.device_combo.currentTextChanged.connect(self.on_device_changed)
        self.device_combo.setFixedHeight(26)

        # Label hugs text, combo expands
        left_layout.addWidget(lbl_device, 0)
        left_layout.addWidget(self.device_combo, 1)

        # --- UI-Settings-Right (Volume) ---
        right_widget = QWidget()
        right_layout = QHBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        lbl_volume = QLabel("Volume:")
        self.volume_bar = QProgressBar()
        self.volume_bar.setObjectName("volumeBar")
        self.volume_bar.setFixedHeight(22)
        self.volume_bar.setRange(0, 100)

        right_layout.addWidget(lbl_volume, 0)
        right_layout.addWidget(self.volume_bar, 1)

        # --- Combine left and right halves equally ---
        setting_layout.addWidget(left_widget, 0, 0)
        setting_layout.addWidget(right_widget, 0, 1)
        setting_layout.setColumnStretch(0, 1)  # left = 50%
        setting_layout.setColumnStretch(1, 1)  # right = 50%

        main_layout.addWidget(setting_group)



        # --- UI-Transcription-Output ---
        output_group = QGroupBox("Transcription Output")
        output_layout = QVBoxLayout(output_group)
        self.transcript_display = QTextEdit()
        self.transcript_display.setFont(QFont("Consolas", 10))
        self.transcript_display.setPlaceholderText("Transcription will appear here...")
        output_layout.addWidget(self.transcript_display)
        main_layout.addWidget(output_group)

        # --- UI-Btn ---
        control_layout = QHBoxLayout()
        self.start_button = QPushButton("▶ Start")
        self.start_button.clicked.connect(self.startTranscription)
        control_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("■ Stop")
        self.stop_button.clicked.connect(self.stopTranscription)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.stop_button)

        self.save_button = QPushButton("💾 Save")
        self.save_button.clicked.connect(self.saveText)
        control_layout.addWidget(self.save_button)

        self.clear_button = QPushButton("✖ Clear")
        self.clear_button.clicked.connect(self.clearTranscript)
        control_layout.addWidget(self.clear_button)


        main_layout.addLayout(control_layout)

    # Audio ===============================================================
    def populate_devices(self):
        self.device_combo.clear()
        devices = QMediaDevices.audioInputs()
        if not devices:
            self.device_combo.addItem("No input devices found")
            self.device_combo.setEnabled(False)
            return

        for dev in devices:
            self.device_combo.addItem(dev.description())

        default = QMediaDevices.defaultAudioInput()
        idx = self.device_combo.findText(default.description())
        if idx >= 0:
            self.device_combo.setCurrentIndex(idx)
        self.statusBar().showMessage(f"Default mic: {default.description()}")

        self.start_qt_audio_monitor(default)
        self.mic.start(default)

    def on_device_changed(self, name: str):
        """Switch active microphone (Qt-based)."""
        for dev in QMediaDevices.audioInputs():
            if dev.description() == name:
                self.mic.stop()
                self.mic.start(dev)
                self.start_qt_audio_monitor(dev)
                self.statusBar().showMessage(f"Switched to mic: {name}")
                print(f"[INFO] Using device: {name}")
                return
        self.statusBar().showMessage(f"Device '{name}' not found.")

    def start_qt_audio_monitor(self, device):
        """Simple Qt mic monitor for volume bar."""
        try:
            if self.audio_source:
                self.audio_source.stop()

            fmt = QAudioFormat()
            fmt.setSampleRate(16000)
            fmt.setChannelCount(1)
            fmt.setSampleFormat(QAudioFormat.Int16)

            self.audio_source = QAudioSource(device, fmt)
            self.io_device = self.audio_source.start()

            if not self.timer:
                self.timer = QTimer()
                self.timer.timeout.connect(self.update_qt_mic_level)
            self.timer.start(80)

            print(f"[INFO] Mic level monitor started for {device.description()}")
        except Exception as e:
            print(f"[ERROR] start_qt_audio_monitor: {e}")

    def update_qt_mic_level(self):
        if not self.io_device:
            return
        data = self.io_device.readAll()
        if data.size() == 0:
            return
        arr = np.frombuffer(data.data(), dtype=np.int16)
        if arr.size == 0:
            return
        rms = np.sqrt(np.mean(np.square(arr.astype(np.float32)))) / 32768.0
        self.volume_bar.setValue(int(min(100, rms * 100)))



    # Btn ===============================================================
    def startTranscription(self):
        if self.asr_t and self.asr_t.is_alive():
            return
        self.asr_worker.running = True
        if not self.mic.running:
            self.mic.start()
        self.asr_t = threading.Thread(target=self.asr_worker.run, daemon=True)
        self.asr_t.start()

        try:
            self.asr_worker.partial.disconnect()
            self.asr_worker.stable.disconnect()
        except Exception:
            pass
        self.asr_worker.partial.connect(self.showPartial)
        self.asr_worker.stable.connect(self.appendStable)

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.statusBar().showMessage("Transcribing...")

    def stopTranscription(self):
        if hasattr(self.asr_worker, "running"):
            self.asr_worker.running = False
        self.mic.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.statusBar().showMessage("Stopped.")


    def saveText(self):
        text = self.transcript_display.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "No Text", "There is no transcription to save.")
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Transcript", os.path.expanduser("~/transcript.txt"),
            "Text Files (*.txt);;All Files (*)"
        )
        if not filename:
            self.statusBar().showMessage("Save cancelled.")
            return
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(text)
            self.statusBar().showMessage(f"Transcript saved to: {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save file:\n{str(e)}")
    
    def toggleTranscription(self):
        if self.asr_t and self.asr_t.is_alive():
            self.stopTranscription()
        else:
            self.startTranscription()

    def clearTranscript(self):
        self.transcript_display.clear()
        self._partial_block_num = None
        self.statusBar().showMessage("Transcription cleared.")

    # Output Text ===============================================================

    def showPartial(self, text: str):
        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#808080"))
        doc = self.transcript_display.document()
        cursor = self.transcript_display.textCursor()
        if self._partial_block_num is not None:
            block = doc.findBlockByNumber(self._partial_block_num)
            if block.isValid():
                cursor.setPosition(block.position())
                cursor.movePosition(QTextCursor.EndOfBlock, QTextCursor.KeepAnchor)
                cursor.removeSelectedText()
                cursor.mergeCharFormat(fmt)
                cursor.insertText(text.strip())
                block.setUserState(1)
        else:
            cursor.movePosition(QTextCursor.End)
            cursor.mergeCharFormat(fmt)
            cursor.insertText(text.strip())
            block = cursor.block()
            block.setUserState(1)
            self._partial_block_num = block.blockNumber()
        self.transcript_display.moveCursor(QTextCursor.End)

    def appendStable(self, text: str):
        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#000000"))
        doc = self.transcript_display.document()
        cursor = self.transcript_display.textCursor()
        if self._partial_block_num is not None:
            block = doc.findBlockByNumber(self._partial_block_num)
            if block.isValid():
                cursor.setPosition(block.position())
                cursor.movePosition(QTextCursor.EndOfBlock, QTextCursor.KeepAnchor)
                cursor.removeSelectedText()
                cursor.mergeCharFormat(fmt)
                cursor.insertText(text.strip() + "\n")
                block.setUserState(0)
        else:
            cursor.movePosition(QTextCursor.End)
            cursor.mergeCharFormat(fmt)
            cursor.insertText(text.strip() + "\n")
        self._partial_block_num = None
        self.transcript_display.moveCursor(QTextCursor.End)

    # Close
    def closeEvent(self, event):
        try:
            if self.audio_source:
                self.audio_source.stop()
            if self.timer:
                self.timer.stop()
            self.stopTranscription()
        except Exception:
            pass
        super().closeEvent(event)
