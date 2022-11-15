import torch
from torch import nn, Tensor
from mtr.modules.ops import Transformer, Res2DMaxPoolModule

class ResNet(nn.Module):
    def __init__(self,
                audio_representation,
                spec_type
        ):
        super(ResNet, self).__init__()
        # Input preprocessing
        self.audio_representation = audio_representation
        self.spec_type = spec_type
        # Input embedding
        self.input_bn = nn.BatchNorm2d(1)
        self.layer1 = Res2DMaxPoolModule(1, 128, pooling=(2, 2))
        self.layer2 = Res2DMaxPoolModule(128, 128, pooling=(2, 2))
        self.layer3 = Res2DMaxPoolModule(128, 128, pooling=(2, 2))
        self.layer4 = Res2DMaxPoolModule(128, 128, pooling=(2, 2))
        self.layer5 = Res2DMaxPoolModule(128, 128, pooling=(2, 2))
        self.layer6 = Res2DMaxPoolModule(128, 128, pooling=(2, 2))
        self.layer7 = Res2DMaxPoolModule(128, 128, pooling=(2, 2))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.to_latent = nn.Identity()
        
    def forward(self, x):
        if self.spec_type == "mel":
            spec = self.audio_representation.melspec(x)
            spec = spec.unsqueeze(1)
        elif self.spec_type == "stft":
            spec = None
        spec = self.input_bn(spec) # B x L x D
        out = self.layer1(spec)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out) # B x D x F x T
        out = self.avg_pool(out)
        h_audio = self.to_latent(out.squeeze(-1).permute(0,2,1)) # B x 1 X D
        return h_audio

class MusicTransformer(nn.Module):
    def __init__(self,
                audio_representation,
                frontend, 
                audio_rep,
                is_vq=False, 
                dropout=0.1, 
                attention_ndim=256,
                attention_nheads=8,
                attention_nlayers=4,
                attention_max_len=512
        ):
        super(MusicTransformer, self).__init__()
        # Input preprocessing
        self.audio_representation = audio_representation
        self.audio_rep = audio_rep
        # Input embedding
        self.frontend = frontend
        self.is_vq = is_vq
        self.vq_modules = None
        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, attention_max_len + 1, attention_ndim))
        self.cls_token = nn.Parameter(torch.randn(attention_ndim))
        # transformer
        self.transformer = Transformer(
            attention_ndim,
            attention_nlayers,
            attention_nheads,
            attention_ndim // attention_nheads,
            attention_ndim * 4,
            dropout,
        )
        self.to_latent = nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.audio_rep == "mel":
            spec = self.audio_representation.melspec(x)
            spec = spec.unsqueeze(1)
        elif self.audio_rep == "stft":
            spec = None
        h_audio = self.frontend(spec) # B x L x D
        if self.is_vq:
            h_audio = self.vq_modules(h_audio)
        cls_token = self.cls_token.repeat(h_audio.shape[0], 1, 1)
        h_audio = torch.cat((cls_token, h_audio), dim=1)
        h_audio += self.pos_embedding[:, : h_audio.size(1)]
        h_audio = self.dropout(h_audio)
        z_audio = self.transformer(h_audio)
        return z_audio
