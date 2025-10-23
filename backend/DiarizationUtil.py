from resemblyzer import VoiceEncoder
import numpy as np

class DiarizationUtil:
    def __init__(self,cfg_file):
        self.encoder = VoiceEncoder()                           # pretrained voice encoder (Resemblyzer)
        self.embedding = []                                     # list of speaker embedding for detected speaker 
        self.speaker_id = []                                    # list of speaker ids
        self.threshold = cfg_file["speaker_threshold"]          # cosine similarity threshold for same speaker
        self.next_id = 1                                        #counter for assigning new speaker 
    
    def identify(self, audio_data: np.ndarray)->str:
        embedding = self.encoder.embed_utterance(audio_data) #compute embedding for audio data 

        if not self.embedding: #first speaker detected - list empty 
            #add data of first speaker
            self.embedding.append(embedding)
            self.speaker_id.append(f"Speaker {self.next_id}")
            self.next_id += 1
            return self.speaker_id[-1]
        
        #compute cosine similarity
        cal_lst = [np.dot(embedding, e) for e in self.embedding]
        best_idx = int(np.argmax(cal_lst))
        if cal_lst[best_idx] > self.threshold: #find best index within threashold
            return self.speaker_id[best_idx]
        
        else: #if new speaker been detected 
            self.embedding.append(embedding)
            self.speaker_id.append(f"Speaker {self.next_id}")
            self.next_id += 1  
            return self.speaker_id[-1]