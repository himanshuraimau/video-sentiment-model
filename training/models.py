import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models as vision_models


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
        self.backbone = vision_models.r3d_18(pretrained=True)
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        num_fts = self.backbone.fc.in_features
        
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_fts, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, x):
        # [batch_size, num_frames, channels, height, width] -> [batch_size , channels ,num_frames , height, width]
        x = x.transpose(1, 2)
        return self.backbone(x)
    
class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
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
        