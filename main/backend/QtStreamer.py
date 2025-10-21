from PySide6.QtMultimedia import QAudioSource, QAudioFormat, QMediaDevices
from PySide6.QtCore import QObject, QTimer, Signal
import numpy as np
import queue


class QAudioStreamer(QObject):

    frame_ready = Signal(np.ndarray)

    def __init__(self, cfg):
        super().__init__()
        self.sample_rate = cfg["sample_rate"]
        self.block_ms = cfg["block_ms"]
        self.block_samples = int(self.sample_rate * self.block_ms / 1000)
        self.channel = cfg.get("channel", 1)

        self.queue = queue.Queue(maxsize=cfg["max_queue"])
        self.queue_timeout = cfg["queue_timeout"]

        self.audio_source = None
        self.io_device = None
        self.timer = None
        self.running = False

    def start(self, device=None):
        fmt = QAudioFormat()
        fmt.setSampleRate(self.sample_rate)
        fmt.setChannelCount(self.channel)
        fmt.setSampleFormat(QAudioFormat.Int16)

        if not device:
            device = QMediaDevices.defaultAudioInput()

        if self.audio_source:
            self.audio_source.stop()

        self.audio_source = QAudioSource(device, fmt)
        self.io_device = self.audio_source.start()

        if not self.timer:
            self.timer = QTimer(self)
            self.timer.timeout.connect(self._read_audio)

        self.timer.start(self.block_ms)
        self.running = True
        print(f"[INFO] QAudioStreamer started on: {device.description()}")

    def stop(self):
        if self.audio_source:
            self.audio_source.stop()
        if self.timer:
            self.timer.stop()
        self.running = False
        print("[INFO] QAudioStreamer stopped.")

    def _read_audio(self):
        if not self.io_device:
            return
        data = self.io_device.readAll()
        if data.size() == 0:
            return

        pcm_bytes = data.data()
        pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
        pcm_f32 = pcm.astype(np.float32) / 32768.0

        try:
            self.queue.put_nowait(pcm_bytes)
            self.frame_ready.emit(pcm_f32)
        except queue.Full:
            pass

    def getFrame(self):
        try:
            return self.queue.get(timeout=self.queue_timeout)
        except queue.Empty:
            return None
