#Vad Util = Voice Detection Utol
import yaml
import webrtcvad

class VadUtils:
    def __init__(self,cfg_file):
        self.vad = webrtcvad.Vad(cfg_file["aggressiveness"])
        self.sample_rate = cfg_file["sample_rate"]

    def isSpeech(self,frame_bytes:bytes)->bool: #check if instance is speech 
        try:
            return self.vad.is_speech(frame_bytes,self.sample_rate)
        except Exception:
            return True