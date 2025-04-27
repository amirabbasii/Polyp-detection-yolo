import torch.nn as nn
from ultralytics import YOLO
from utils import create_data_directories

class LightweightAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=8):
        super().__init__()
        self.channels = channels
        reduced_channels = max(channels // reduction_ratio, 32)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_attention = nn.Sequential(
            nn.Linear(channels, reduced_channels),
            nn.SiLU(),
            nn.Linear(reduced_channels, channels),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.shape
        channel_att = self.avg_pool(x).view(b, c)
        channel_att = self.channel_attention(channel_att).view(b, c, 1, 1)
        spatial_att = self.spatial_attention(x)
        return x * channel_att * spatial_att

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = LightweightAttention(channels)
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.SiLU()

    def forward(self, x):
        identity = x
        x = self.attention(x)
        x = self.norm(x)
        x = self.act(x + identity)
        return x

def add_attention_to_model(model):
    backbone = model.model.model
    attention_indices = [9, 12]
    for idx in attention_indices:
        if idx < len(backbone):
            module = backbone[idx]
            if hasattr(module, 'conv'):
                channels = module.conv.out_channels
                attention = AttentionBlock(channels)
                backbone[idx] = nn.Sequential(module, attention)
    return model
