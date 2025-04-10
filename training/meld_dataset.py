from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
import os
import cv2
import numpy as np
import torch

class MELDDataset(Dataset):
    
    def __init__(self,csv_path,video_dir):
        self.data = pd.read_csv(csv_path)
        self.video_dir = video_dir
        
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Anger, Disgust, Sadness, Joy, Neutral, Surprise and Fear
        self.emotion_map = {'anger': 0,'disgust': 1,'sadness': 2,'joy': 3,'neutral': 4,'surprise': 5,'fear': 6 }
        
        self.sentiment_map = {'positive': 0,'negative': 1,'neutral': 2 }
    
    def __load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        try:
            if not cap.isOpened():
                raise ValueError(f"Could not open video file {video_path}")
           # try and read first frame to validate first
            ret, frame = cap.read()
            if not ret or frame is None:
                raise ValueError(f"Could not read video file {video_path}")
            # read the rest of the frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            while (len(frames))<30 and cap.isOpened():
                ret,frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.resize(frame, (224, 224))
                frame = frame.astype('float32') / 255.0 # [255,128,0] -> [1,0.5,0]
                frames.append(frame)    
            
        except Exception as e:
            raise ValueError(f"Error loading video frames: {e}")
        finally:
            cap.release()
        
        if (len(frames)==0):
            raise ValueError(f"No frames extracted from video file {video_path}")
        
        # pad or truncate frames to 30
        if len(frames)<30:
            frames += [np.zeros_like(frames[0])] *(30-len(frames))
        else:
            frames = frames[:30]
        
        # Befor the permute: [frames, height, width, channels]
        # After the permute: [frames, channels, height, width]
        
        return torch.FloatTensor(np.array(frames)).permute(0,3,1,2)
           
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row  =  self.data.iloc[idx]
        video_filename = f"""dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4""" 
        path = os.path.join(self.video_dir, video_filename) 
        video_path_exists = os.path.exists(path)
        
        if video_path_exists== False:
            raise FileNotFoundError(f"Video file {video_filename} not found in {self.video_dir}")
        
        text_inputs = self.tokenizer(row['Utterance'], 
                                     padding='max_length', 
                                     truncation=True, 
                                     max_length=128, 
                                     return_tensors="pt")
        video_frames = self.__load_video_frames(path)
        
        print(video_frames)
        

if __name__ == "__main__":
    meld = MELDDataset('dataset/dev/dev_sent_emo.csv',
                       'dataset/dev/dev_splits_complete')
    
    print(meld[0])