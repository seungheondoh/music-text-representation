import torch
import torchaudio
import math
import numpy as np
import random
from torch import Tensor, nn
from mtr.modules.head import TripletHead, SyncFunction

class TripletModel(nn.Module):
    def __init__(self, audio_encoder, text_encoder, text_type, audio_dim, text_dim, mlp_dim, margin):
        super(TripletModel, self).__init__()
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.text_type = text_type
        self.head = TripletHead(margin=margin)
        self.audio_projector = nn.Sequential(nn.LayerNorm(audio_dim), nn.Linear(audio_dim, mlp_dim, bias=False))
        self.text_projector =  nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, mlp_dim, bias=False))
        self.audio_encoder.train()
        self.text_encoder.train()
        self.a_latent = nn.Identity()
        self.t_latent = nn.Identity()

    def forward(self, binary, audio, text, text_mask=None):
        h_audio = self.encode_audio(audio)
        if self.text_type == "bert":
            h_text = self.encode_bert_text(text, text_mask)
        elif self.text_type == "glove":
            h_text = self.encode_glove_text(text)
        h_audio_at, h_text_at, h_neg_text_at = self.triplet_sampling(h_audio, h_text, binary)
        h_text_ta, h_audio_ta, h_neg_audio_ta = self.triplet_sampling(h_text, h_audio, binary)
        at_loss = self.head(h_audio_at, h_text_at, h_neg_text_at)
        ta_loss = self.head(h_text_ta, h_audio_ta, h_neg_audio_ta)
        loss = (at_loss + ta_loss) / 2
        return loss
        
    def encode_audio(self, audio):
        # audio = (Batch x Length x Dim)
        audio_emb = self.audio_encoder(audio)
        h_audio = self.a_latent(audio_emb[:,0,:])
        z_audio = self.audio_projector(h_audio)
        return z_audio

    def encode_bert_text(self, text, text_mask):
        text_emb = self.text_encoder(input_ids=text, attention_mask=text_mask)
        h_text = self.t_latent(text_emb['last_hidden_state'][:,0,:])
        z_text = self.text_projector(h_text)
        return z_text

    def encode_glove_text(self, text_emb): 
        h_text = self.t_latent(text_emb)
        z_text = self.text_projector(h_text)
        return z_text

    def triplet_sampling(self, anchor_emb, positive_emb, binary):
        if torch.distributed.get_world_size() > 1:
            anchor_emb = SyncFunction.apply(anchor_emb)
            positive_emb = SyncFunction.apply(positive_emb)
            binary = SyncFunction.apply(binary)
        num_batch = len(anchor_emb)
        # get distance weights
        anchor_norm = torch.nn.functional.normalize(anchor_emb, dim=1)
        positive_norm = torch.nn.functional.normalize(positive_emb, dim=1)
        dot_sim = torch.matmul(anchor_norm, positive_norm.T)
        weights = (dot_sim + 1) / 2
        # masking => high value: easy negative, low vale: hard negative
        sum_= binary.sum(dim=1, keepdim=True)
        mask = sum_ - torch.matmul(binary, binary.T)  # 
        hard_pos = 1 / mask
        multi_label_mask = torch.nan_to_num(hard_pos, posinf=.0)
        masked_weights = weights * multi_label_mask
        # sampling
        index_array = torch.arange(num_batch)
        negative_ix = [random.choices(index_array, weights=masked_weights[i], k=1)[0].item() for i in range(num_batch)]
        negative_emb = positive_emb[negative_ix]
        return anchor_emb, positive_emb, negative_emb
    