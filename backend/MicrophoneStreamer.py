from PySide6.QtMultimedia import QAudioSource, QMediaDevices, QAudioFormat
from PySide6.QtCore import QTimer, QObject, Signal
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MicrophoneStreamerQt(QObject):
    
    def __init__(self, sample_rate=16000, block_ms=20):
        super().__init__()
        self.sample_rate = sample_rate
        self.block_ms = block_ms
        self.device = QMediaDevices.defaultAudioInput()
        self.audio_source = None
        self.io_device = None
        self.timer = None
        self.last_frame = None
        self.is_running = False

        # Initialize audio
        self._setup_audio()

    def _setup_audio(self):
        try:
            # audio format
            fmt = QAudioFormat()
            fmt.setSampleRate(self.sample_rate)
            fmt.setChannelCount(1)
            fmt.setSampleFormat(QAudioFormat.Int16)

            # setup audio source
            self.audio_source = QAudioSource(self.device, fmt)
            self.io_device = self.audio_source.start()

            # setup polling timer
            self.timer = QTimer()
            self.timer.timeout.connect(self.read_audio_data)
            self.timer.start(self.block_ms)
            
            self.is_running = True
            
        except Exception as e:
            logger.error(f"Failed to setup microphone: {str(e)}")
            raise

    def read_audio_data(self):
        if not self.is_running or not self.io_device:
            return
            
        try:
            data = self.io_device.readAll()
            if data.size() == 0:
                return

            # convert bytes to numpy array
            samples = np.frombuffer(data.data(), dtype=np.int16)
            self.last_frame = samples

            if hasattr(self, '_debug_count'):
                self._debug_count += 1
            else:
                self._debug_count = 0
                
            if self._debug_count % 1000 == 0:  
                rms = np.sqrt(np.mean(samples**2))
                logger.debug(f"Debug- {rms:.2f}, samples: {len(samples)}")
                
        except Exception as e:
            logger.error(f"Error reading audio data: {str(e)}")

    def getFrame(self):
        return self.last_frame

    def stop(self):
        try:
            self.is_running = False
            
            if self.timer:
                self.timer.stop()
                self.timer.deleteLater()
                self.timer = None
                
            if self.audio_source:
                self.audio_source.stop()
                self.audio_source.deleteLater()
                self.audio_source = None
                
            if self.io_device:
                self.io_device = None
                
            logger.info("Microphone streamer stopped")
            
        except Exception as e:
            logger.error(f"Error stopping microphone streamer: {str(e)}")

    def is_active(self):
        return self.is_running and self.audio_source is not None


class MicrophoneStreamerWrapper:
    
    def __init__(self, audio_source, io_device):
        self.audio_source = audio_source
        self.io_device = io_device
        self.last_frame = None
        self.is_running = True
        
    def getFrame(self):
        if not self.is_running or not self.io_device:
            return None
            
        try:
            data = self.io_device.readAll()
            if data.size() == 0:
                return None
                
            # Convert bytes to numpy array
            samples = np.frombuffer(data.data(), dtype=np.int16)
            self.last_frame = samples
            return samples
        except Exception as e:
            print(f"Error reading audio frame: {e}")
            return None
    
    def is_active(self):
        return self.is_running and self.audio_source is not None
    
    def stop(self):
        self.is_running = False
