import torch
import torchaudio
import math
import numpy as np
from torch import Tensor, nn
from mtr.modules.head import ClsHead

class ClassificationModel(nn.Module):
    def __init__(self, audio_encoder, audio_dim, mlp_dim, num_classes):
        super(ClassificationModel, self).__init__()
        self.audio_encoder = audio_encoder
        self.audio_dim = audio_dim
        self.audio_projector = nn.Sequential(nn.LayerNorm(audio_dim), nn.Linear(audio_dim, mlp_dim, bias=False))
        self.head = ClsHead(in_channels=mlp_dim, num_classes=num_classes)
        self.audio_encoder.train()
        self.a_latent = nn.Identity()

    def forward(self, audio, label):
        z_audio = self.encode_audio(audio)
        loss = self.head(z_audio, label)
        return loss
    
    def forward_eval(self, audio):
        # audio = (Batch x Length x Dim)
        z_audio = self.encode_audio(audio)
        output = self.head.fc_cls(z_audio)
        logit = self.head.sigmoid(output)
        return logit
        
    def encode_audio(self, audio):
        # audio = (Batch x Length x Dim)
        audio_emb = self.audio_encoder(audio)
        h_audio = self.a_latent(audio_emb[:,0,:])
        z_audio = self.audio_projector(h_audio)
        z_audio = nn.functional.normalize(z_audio, dim=1)
        return z_audio