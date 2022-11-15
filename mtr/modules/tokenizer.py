import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mtr.modules.ops import Res2DMaxPoolModule

class SpecPatchEmbed(nn.Module):
    """ 2D spectrogram to Patch Embedding
    """
    def __init__(self, f_size=128, t_size=1024, p_w=16, p_h=16, in_chans=1, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        self.spec_size = (f_size, t_size)
        self.patch_size = (p_w, p_h)
        self.grid_size = (self.spec_size[0] // self.patch_size[0], self.spec_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class ResFrontEnd(nn.Module):
    """
    After the convolution layers, we flatten the time-frequency representation to be a vector.
    mix_type : cf -> mix channel and frequency dim
    mix_type : ft -> mix frequency and time dim
    """
    def __init__(self, input_size ,conv_ndim, attention_ndim, mix_type="cf",nharmonics=1):
        super(ResFrontEnd, self).__init__()
        self.mix_type = mix_type
        self.input_bn = nn.BatchNorm2d(nharmonics)
        self.layer1 = Res2DMaxPoolModule(nharmonics, conv_ndim, pooling=(2, 2))
        self.layer2 = Res2DMaxPoolModule(conv_ndim, conv_ndim, pooling=(2, 2))
        self.layer3 = Res2DMaxPoolModule(conv_ndim, conv_ndim, pooling=(2, 2))
        self.layer4 = Res2DMaxPoolModule(conv_ndim, conv_ndim, pooling=(2, 2))
        F,T = input_size
        self.ntime = T // 2 // 2 // 2 // 2
        self.nfreq = F // 2 // 2 // 2 // 2
        if self.mix_type == "ft":
            self.fc_ndim = conv_ndim
        else:
            self.fc_ndim = self.nfreq * conv_ndim
        self.fc = nn.Linear(self.fc_ndim, attention_ndim)

    def forward(self, hcqt):
        # batch normalization
        out = self.input_bn(hcqt)
        # CNN
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # permute and channel control
        b, c, f, t = out.shape
        if self.mix_type == "ft":
            out = out.contiguous().view(b, c, -1)  # batch, channel, tf_dim
            out = out.permute(0,2,1) # batch x length x dim
        else:
            out = out.permute(0, 3, 1, 2)  # batch, time, conv_ndim, freq
            out = out.contiguous().view(b, t, -1)  # batch, length, hidden
        out = self.fc(out)  # batch, time, attention_ndim
        return out