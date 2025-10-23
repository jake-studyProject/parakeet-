import yaml
from abc import ABC, abstractmethod
import numpy as np

#asr model imports 
import onnx_asr

class AsrModel(ABC):  #Uniform Interface for ASR Models 
    @abstractmethod
    def transcribe(self,audio_sample:np.ndarray)->str:
        # transcribe audio file given path
        pass
    @abstractmethod
    def __str__(self):
        #srting representation 
        pass

class NvidiaParakeet(ABC):
    def __init__(self,cfg_file):
        self.model_name = cfg_file["model_name"]
        self.asr_model = onnx_asr.load_model(cfg_file["model_name"],cfg_file["model_dir"])

    def transcribe(self,audio_sample:np.ndarray)->str:
        if len(audio_sample) == 0:
            return ""
        else:
            #step 1: convert audio to numpy array 
            decode_audio = audio_sample.astype(np.float32, copy=False) 
            #step 2: 
            text = self.asr_model.recognize(decode_audio)
            return text 

    def __str__(self):
        return self.model_name

        
