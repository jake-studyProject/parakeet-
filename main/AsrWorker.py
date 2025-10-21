import yaml
import time
import numpy as np
import collections

from datetime import datetime

import MicrophoneStreamer as ms 
import VadUtils as vadu 
import AsrModel as am 
import DiarizationUtil as du

class ParakeetAsrWorker:
    def __init__(self,mic:ms,asr_model:am,vad:vadu,diarize:du,cfg_file):
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

    def run(self):
        float_buf = collections.deque()                 #store audio for queue
        buf_samples = 0                                 #num of sample in buff
    
        in_speech = False                               #flag for speech 
        last_speech_ts = time.time()                    #timestamp for last speech
        last_flush_ts = time.time()                     #timestamp for last audio sent to asr 

        while self.running:
            frame = self.mic.getFrame() #get audio frame from mic 

            #Handle Silence-Timeout 
            if frame is None: 

                #clean up buffer 
                if in_speech and (time.time() - last_speech_ts) > 0.6:
                    text = self.flushTotext(float_buf,force=True) 

                    if text.strip():
                        self.final_segments.append(text.strip() + " ")
                        self.trimHistoryToBudget()
                    
                    float_buf.clear()
                    buf_samples = 0
                    in_speech = False

                continue

            #Check if speech is detected + turn audio into array 
            talking = self.vad.isSpeech(frame)
            block_int16 = np.frombuffer(frame, dtype=np.int16)
            block_float = block_int16.astype(np.float32) / 32767.0

            
            if talking:
                in_speech = True
                last_speech_ts = time.time()
                #update buff + buff size
                float_buf.append(block_float)
                buf_samples += block_float.size


                #Trim buffer if too big
                if buf_samples > self.max_samples:
                    while buf_samples > self.keep_left and float_buf:
                        left = float_buf.popleft()
                        buf_samples -= left.size

                if (time.time() - last_flush_ts) > self.cfg_file["partial_refresh_sec"] or (len(float_buf) * self.block_ms / 1000.0 ) >= self.chunk_max_sec:
                    text = self.flushTotext(float_buf,force=False)

                    if text:
                        self.current_partial = text
                        last_flush_ts = time.time()
                        
            else:
                #Handle end of speech 
                if in_speech and (time.time() - last_speech_ts) > 0.3:
                    text,speaker,ts= self.flushTotext(float_buf,force=True)
                    if text.strip():
                        self.final_segments.append((ts,speaker,text.strip() + " ")) #timestamp and text tupple 
                        self.trimHistoryToBudget()
                            
                    in_speech = False
                    float_buf.clear()
                    buf_samples = 0
                        








