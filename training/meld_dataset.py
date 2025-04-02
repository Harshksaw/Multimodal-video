from torch.utils.data import Dataset
import pandas as pd
import os
from transformers import AutoTokenizer
import numpy as np
import cv2
import torch
import touchaudio


class  MELDDataset(Dataset):

    
    def __init__(self, csv_path , video_dir):
        self.data= pd.read_csv(csv_path)
        self.video_dir = video_dir
        self.tokeninzer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        self.emotion_map ={
            'anger':0,
            'disgust':1,
            'fear':2,
            'joy':3,
            'neutral':4,
            'sadness':5,
            'surprise':6
        }
        self.sentiment_map = {
            'positive':0,
            'negative':1,
            'neutral':2
        }
        
        

        
    def  _load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames =[]
        
        try:
            if not cap.isOpened():
                raise FileNotFoundError(f"Video not found at {video_path}")
            
            ret , frame = cap.read()
            #try and read first frame to validate video
            if not ret or frame is None:
                raise ValueError(f"Error loading video at {video_path}")
            
            # Reset index to not CAP
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while len(frames)< 30 and cap.isOpened():

                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (224,224))
                
                #normalize frame
                frame = frame/255.0
                frames.append(frame)
                
                
                
                
            
        except Exception as e:
            raise ValueError(f"Error loading video at {video_path}")
        finally:
            cap.release()
            
        if (len(frames)  == 0):
            raise ValueError(f"No frames could be extracted")
        
        #pad or truncate
        if(len(frames) < 30):
            frames += [np.zeros_like(frames[0]) * (30-len(frames))]
        else:
            frames = frames[:30]
            
            
            
            
            #Befor permute : [frames, height, width, channels]
            #After permute : [frames, channels, height, width]
            
        return torch.FloatTensor(np.array(frames)).permute(0,3,1,2)
    
            
            
            
    def _extract_audio_features(self, video_path):   
# Generate the output audio file path by replacing the .mp4 extension with .wav
        audio_path = video_path.replace('.mp4', '.wav')

        try:
    # Use ffmpeg to extract audio from the video file
    # -vn: Disable video processing
    # -acodec pcm_s16le: Use PCM 16-bit little-endian codec
    # -ac 1: Convert audio to mono
    # -ar 16000: Set audio sample rate to 16,000 Hz
    # Suppress ffmpeg output and errors using subprocess.DEVNULL
        subprocess.run(['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000', audio_path], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
    # Raise an error if audio extraction fails
     raise ValueError(f"Error extracting audio from {video_path}")
 
 waveform, sample_rate = torchaudio.load()
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        
    mel_spectogram = torchaudio.transforms.MelSpectorgram(
        sample_rate = 16000,
        n_mels= 64,
        n_fft= 1024
        hop_length= 512
        
        
    )
    mel_spec = mel_spectrogram(waveform)
    
    #Normalize 
    mel_spec = (mel_spec - mel_spect.mean())/mel_spec.std()
    
    if mel_spec.size(2)< 300:
        padding = 300 - mel_spec.size(2)
        mel_spec = torch.nn.functional.pad(mel_spec, (0 ,padding))
    else:
        mel_spec = mel_spec[:,:,:300]
        
    except subprocessCalledProcessError as e:
        raise ValueError(f"Audio extractio error":{str(e)})
    except Exception as e:
        raise ValueError(f"Audio Error:{str(e)}")
    finally: 
        if os.path.exists(audio_path):
            os.remove(audio_path)
            
        
            
            
            
    def __len__(self):
        
        return len(self.data)
    
    def __getitem__(self, idx):
        # print(self.data.iloc[idx])
        row = self.data.iloc[idx]
        
        video_filename = f"""dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"""
        
        path = os.path.join(self.video_dir, video_filename)
        
        video_path = os.path.exists(path)
        
        
        if(video_path == False):
            # print(f"Video not found at {path}")
            raise FileNotFoundError("No video found for fileanme: {path}")
        

        print("Video found at ", path)
        
        text_inputs = self.tokeninzer(row['Utterance'], return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        
        # print(text_inputs)
        
        #load video frames
        video_frames = self._load_video_frames(path)
        print(video_frames)
        
if __name__ == "__main__":
    meld = MELDDataset('../dataset/dev/dev_sent_emo.csv', '../dataset/dev/dev_splits_complete')
    print(meld[0])