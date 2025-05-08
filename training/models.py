import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models as vision_models

from meld_dataset import MELDDataset


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        
        for param in self.bert.parameters():
            param.requires_grad = False
        
        self.projection = nn.Linear(768, 128)
        
    def forward(self, input_ids, attention_mask):
        # Extract bert embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        #use [CLS] token representation
        pooled_output = outputs.pooler_output
        
        return self.projection(pooled_output)

class VideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = vision_models.video.r3d_18(weights=vision_models.video.R3D_18_Weights.DEFAULT)
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        num_fts = self.backbone.fc.in_features
        
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_fts, 128),
            nn.ReLU(),
            nn.Dropout(0.2) # Project down to 128 to match other modalities
        )
        
    def forward(self, x):
        # [batch_size, num_frames, channels, height, width] -> [batch_size , channels ,num_frames , height, width]
        x = x.transpose(1, 2)
        return self.backbone(x)
    
class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Input channels should match mel spectrogram features (128)
        self.conv_layers = nn.Sequential(
            # Low level features
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Higher level features
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        
        for param in self.conv_layers.parameters():
            param.requires_grad = False
        
        self.projection = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, x): 
        x = x.squeeze(1)
        features = self.conv_layers(x)
        # features output shape: [batch_size, 128, 1]
        return self.projection(features.squeeze(-1))

class MultimodalSentimentModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.text_encoder = TextEncoder()
        self.video_encoder = VideoEncoder()
        self.audio_encoder = AudioEncoder()
        
        # Fusion Layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Classification Heads
        self.emo_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 7)
        )
        
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3) 
        )
    def forward(self,text_inputs,video_frames,audio_features):
        text_features = self.text_encoder(
            text_inputs['input_ids'],
            text_inputs['attention_mask']
        )
        
        video_features = self.video_encoder(video_frames)
        
        audio_features = self.audio_encoder(audio_features)
        
        # Concatenate multimodal features
        combined_features = torch.cat((text_features, video_features, audio_features), dim=1)
        # Pass through fusion layer
        fused_features = self.fusion_layer(combined_features)
        
        emotion_output = self.emo_classifier(fused_features)
        sentiment_output = self.sentiment_classifier(fused_features)
        
        return {
            'emotions': emotion_output,
            'sentiment': sentiment_output
        }   

if __name__ == "__main__":
    
    dataset  = MELDDataset(
        '../dataset/train/train_sent_emo.csv',
        '../dataset/train/train_splits_complete',
    )
    
    sample = dataset[0]
    
    model = MultimodalSentimentModel()
    model.eval()
    
    text_inputs = {
        'input_ids': sample['text_inputs']['input_ids'].unsqueeze(0),
        'attention_mask': sample['text_inputs']['attention_mask'].unsqueeze(0)
    }
    
    video_frames = sample['video_frames'].unsqueeze(0)
    audio_features = sample['audio_features'].unsqueeze(0)
    
    with torch.inference_mode():
        outputs = model(text_inputs, video_frames, audio_features)
        
        emotion_probs = torch.softmax(outputs['emotions'], dim=1)[0]
        sentiment_probs = torch.softmax(outputs['sentiment'], dim=1)[0]
    
    emotion_map = {
        0:'anger',1:'disgust',2:'sadness',3:'joy',4:'neutral',5:'surprise',6:'fear'
    }
        
    sentiment_map = {
        0:'positive',1:'negative',2:'neutral'
    }
    
    print("Emotion Predictions:")
    for i, prob in enumerate(emotion_probs):
        print(f"{emotion_map[i]}: {prob.item():.4f}")
    print("\nSentiment Predictions:")
    for i, prob in enumerate(sentiment_probs):
        print(f"{sentiment_map[i]}: {prob.item():.4f}")
    print()