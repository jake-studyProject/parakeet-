import yaml
import time
import numpy as np
import collections

from datetime import datetime

from . import MicrophoneStreamer as ms 
from . import VadUtils as vadu 
from . import AsrModel as am 
from . import DiarizationUtil as du

from PySide6.QtCore import QObject, Signal


class ParakeetAsrWorker(QObject):
    stable = Signal(str)
    partial = Signal(str)

    def __init__(self,mic:ms,asr_model:am,vad:vadu,diarize:du,cfg_file):
        super().__init__()
        self.mic = mic 
        self.asr_model = asr_model
        self.vad = vad 
        self.diarize = diarize

        self.cfg_file = cfg_file

        #result boundaries 
        self.final_char_budget = int(cfg_file["max_history_sec"]*18)                                 #character limit 
        self.max_samples = int(cfg_file["window_sec"] * cfg_file["sample_rate"])                     #maximum # of sample limit
        self.keep_left = int(cfg_file["context_overlap_sec"] * self.cfg_file["sample_rate"])    #target size to trim buffer
        self.chunk_max_sec = cfg_file["chunk_max_sec"]
        self.block_ms = cfg_file["block_ms"]

        #live results and flag 
        self.final_segments = collections.deque(maxlen=9999)
        self.current_partial = ""
        self.running = True

    def flushTotext(self,buf,force=False): 
        
        if not buf: #if buffer empty 
            return ""
        
        samples = np.concatenate(list(buf), axis=0)
        dur = samples.size / self.cfg_file["sample_rate"]

        if not force and dur < self.cfg_file["chunk_min_sec"]:
            return ""
        
        ts = datetime.now().strftime("%H:%M:%S")
        text = self.asr_model.transcribe(samples) 
        speaker = self.diarize.identify(samples)

        return text,speaker,ts
    
    def trimHistoryToBudget(self):
        total = sum(len(s) for s in self.final_segments)
        while total > self.final_char_budget and self.final_segments:
            total -= len(self.final_segments.popleft())


# as the QaudioSource always deliver bytes even silence frame it just no working

    def run(self):
        float_buf = collections.deque()                 # store audio frames
        buf_samples = 0                                 # num of samples in buffer

        in_speech = False                               # track if we're currently in speech
        last_speech_ts = time.time()                    # timestamp of last detected speech
        last_flush_ts = time.time()                     # timestamp for partial flush

        # while self.running:
        #     frame = self.mic.getFrame()

        #     # Handle mic silence (no frame yet)
        #     if frame is None:
        #         time.sleep(0.01)
        #         # flush after longer silence timeout
        #         if in_speech and (time.time() - last_speech_ts) > 0.8:
        #             result = self.flushTotext(float_buf, force=True)
        #             if isinstance(result, tuple):
        #                 text, speaker, ts = result
        #                 if text and text.strip():
        #                     self.final_segments.append((ts, speaker, text.strip() + " "))
        #                     self.trimHistoryToBudget()
        #                     self.stable.emit(f"[{ts}] {speaker}: {text.strip()}")
        #                     print(f"[DEBUG] stable emitted (no-frame): {text.strip()}")
        #             in_speech = False
        #             float_buf.clear()
        #             buf_samples = 0
        #             self.current_partial = ""
        #         continue

        #     # Convert frame
        #     block_int16 = np.frombuffer(frame, dtype=np.int16)
        #     block_float = block_int16.astype(np.float32) / 32767.0

        #     # VAD + fallback energy gate
        #     talking = self.vad.isSpeech(frame)
        #     energy = np.mean(np.abs(block_float))
        #     silence = energy < 0.01

        #     # ===== ACTIVE SPEECH =====
        #     if talking and not silence:
        #         in_speech = True
        #         last_speech_ts = time.time()

        #         float_buf.append(block_float)
        #         buf_samples += block_float.size

        #         # Trim old samples
        #         if buf_samples > self.max_samples:
        #             while buf_samples > self.keep_left and float_buf:
        #                 left = float_buf.popleft()
        #                 buf_samples -= left.size

        #         # Periodic partial flush
        #         if (time.time() - last_flush_ts) > self.cfg_file["partial_refresh_sec"] \
        #             or (len(float_buf) * self.block_ms / 1000.0) >= self.chunk_max_sec:
                    
        #             result = self.flushTotext(float_buf, force=False)
        #             if isinstance(result, tuple):
        #                 text, speaker, ts = result
        #                 if text and text.strip():
        #                     self.current_partial = text
        #                     self.partial.emit(f"[{ts}] {speaker}: {text}")
        #                     print(f"[DEBUG] partial emitted: {text}")
        #                     last_flush_ts = time.time()

        #     # ===== END OF SPEECH =====
        #     else:
        #         silence_dur = time.time() - last_speech_ts
        #         if in_speech and silence_dur > 0.8:  # restore longer timeout
        #             print(f"[DEBUG] Silence {silence_dur:.2f}s -> flushing stable")
        #             result = self.flushTotext(float_buf, force=True)
        #             if isinstance(result, tuple):
        #                 text, speaker, ts = result
        #                 if text and text.strip():
        #                     self.final_segments.append((ts, speaker, text.strip() + " "))
        #                     self.trimHistoryToBudget()
        #                     self.stable.emit(f"[{ts}] {speaker}: {text.strip()}")
        #                     print(f"[DEBUG] stable emitted: {text.strip()}")

        #             in_speech = False
        #             float_buf.clear()
        #             buf_samples = 0
        #             self.current_partial = ""
        #             last_flush_ts = time.time()

        while self.running:
            frame = self.mic.getFrame()

            # Handle mic silence (no frame yet)
            if frame is None:
                time.sleep(0.01)
                # flush after longer silence timeout
                if in_speech and (time.time() - last_speech_ts) > 0.6:
                    result = self.flushTotext(float_buf, force=True)

                    if isinstance(result, tuple):
                        text, speaker, ts = result
                        if text and text.strip():
                            self.final_segments.append((ts, speaker, text.strip() + " "))
                            self.trimHistoryToBudget()
                            self.stable.emit(f"[{ts}] {speaker}: {text.strip()}")
                            print(f"[DEBUG] stable emitted (no-frame): {text.strip()}")
                    float_buf.clear()
                    buf_samples = 0
                    in_speech = False
                    self.current_partial = ""
                continue

            # Convert frame
            talking = self.vad.isSpeech(frame)
            block_int16 = np.frombuffer(frame, dtype=np.int16)
            block_float = block_int16.astype(np.float32) / 32767.0

            # noise control future extensions of optional selection depending on bg noise
            energy = np.mean(np.abs(block_float))
            silence = energy < 0.012


            if talking and not silence:
                in_speech = True
                last_speech_ts = time.time()

                float_buf.append(block_float)
                buf_samples += block_float.size

                #Trim buffer if too big
                if buf_samples > self.max_samples:
                    while buf_samples > self.keep_left and float_buf:
                        left = float_buf.popleft()
                        buf_samples -= left.size

                # Periodic partial flush
                if (time.time() - last_flush_ts) > self.cfg_file["partial_refresh_sec"] \
                    or (len(float_buf) * self.block_ms / 1000.0) >= self.chunk_max_sec:
                    
                    result = self.flushTotext(float_buf, force=False)

                    if isinstance(result, tuple):
                        text, speaker, ts = result
                        if text and text.strip():
                            self.current_partial = text
                            self.partial.emit(f"[{ts}] {speaker}: {text}")
                            print(f"[DEBUG] partial emitted: {text}")
                            last_flush_ts = time.time()

            #End of speech
            else:

                if in_speech and (time.time() - last_speech_ts) > 0.3:
                    result = self.flushTotext(float_buf, force=True)
                    if isinstance(result, tuple):
                        text, speaker, ts = result
                        if text and text.strip():
                            self.final_segments.append((ts, speaker, text.strip() + " "))
                            self.trimHistoryToBudget()
                            self.stable.emit(f"[{ts}] {speaker}: {text.strip()}")
                            print(f"[DEBUG] stable emitted: {text.strip()}")

                    in_speech = False
                    float_buf.clear()
                    buf_samples = 0
                    self.current_partial = ""
                    last_flush_ts = time.time()





