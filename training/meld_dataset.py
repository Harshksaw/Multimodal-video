from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from transformers import AutoTokenizer
import numpy as np
import cv2
import torch
import torchaudio
import subprocess

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
            subprocess.run(
                ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000', audio_path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            # Load the extracted audio file using torchaudio
            waveform, sample_rate = torchaudio.load(audio_path)

            # Resample the audio if the sample rate is not 16,000 Hz
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)

            # Generate a Mel spectrogram from the waveform
            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=64,
                n_fft=1024,
                hop_length=512
            )
            mel_spec = mel_spectrogram(waveform)

            # Normalize the Mel spectrogram
            mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()

            # Pad or truncate the Mel spectrogram to ensure a fixed size
            if mel_spec.size(2) < 300:
                padding = 300 - mel_spec.size(2)
                mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
            else:
                mel_spec = mel_spec[:, :, :300]

            return mel_spec

        except subprocess.CalledProcessError as e:
            # Handle errors during the ffmpeg command execution
            raise ValueError(f"Audio extraction error: {str(e)}")
        except Exception as e:
            # Handle other errors during audio processing
            raise ValueError(f"Audio processing error: {str(e)}")
        finally:
            # Clean up by removing the temporary audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)
        
            
            
            
    def __len__(self):
        
        return len(self.data)
    
    def __getitem__(self, idx):
        
        if isinstance(idx, torch.Tensor):
            
            idx = idx.item()
            
        row = self.data.iloc[idx]
            
        try:
            
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
            # print(video_frames, "video frames")
            
            audio_features = self._extract_audio_features(path)
            # print(audio_features, "audio features")
            #Map sentiment and emotion label
            emotion_label = self.emotion_map[row['Emotion'].lower()]
            sentiment_label = self.sentiment_map[row['Sentiment'].lower()]
            print(emotion_label, "emotion label")
            print(sentiment_label, "sentiment label")
            
            
            return{
                'text_inputs': {
                    'input_ids': text_inputs['input_ids'].squeeze(0),
                    'attention_mask': text_inputs['attention_mask'].squeeze(0),

                    },
                'video_frames': video_frames,
                'audio_features': audio_features,
                'emotion_label': torch.tensor(emotion_label, dtype=torch.long),
                'sentiment_label': torch.tensor(sentiment_label, dtype=torch.long)
            }
        except Exception as e:
            print(f"Error processing index {idx}: {e}")   
            return None
        # finally:
            # Clean up by removing the temporary audio file if it exists
            # if os.path.exists(path):
            #     os.remove(path) 
        
def collate_fn(batch):
    batch = list(filter(None , batch))  # Remove None values
    if len(batch) == 0:
        return {}
    return torch.utils.data.dataloader.default_collate(batch)
        
def prepare_dataloader(train_csv, train_video_dir, dev_csv , dev_video_dir, test_csv , test_video_dir , batch_size = 32):
    train_dataset = MELDDataset(train_csv, train_video_dir)
    dev_dataset = MELDDataset(dev_csv, dev_video_dir)
    test_dataset = MELDDataset(test_csv, test_video_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,collate_fn=collate_fn, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size,collate_fn=collate_fn, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    return train_loader, dev_loader, test_loader



        
if __name__ == "__main__":

    # meld = MELDDataset('../dataset/dev/dev_sent_emo.csv', '../dataset/dev/dev_splits_complete')
    train_loader , dev_loader , test_loader = prepare_dataloader('../dataset/train/train_sent_emo.csv', '../dataset/train/train_splits_complete',
                                                               '../dataset/dev/dev_sent_emo.csv', '../dataset/dev/dev_splits_complete',
                                                               '../dataset/test/test_sent_emo.csv', '../dataset/test/test_splits_complete')
    
    # print(meld[0],'melddataset')
    
    for batch in train_loader:
        if not batch:
            continue  
        print(batch['text_inputs'])
        print(batch['video_frames'].shape)
        print(batch['audio_features'].shape)
        print(batch['emotion_label'])
        print(batch['sentiment_label'])
        break
    
