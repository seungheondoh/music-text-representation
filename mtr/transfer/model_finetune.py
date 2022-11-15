import torch
import torch.nn as nn

class CLSLayer(nn.Module):
    def __init__(self, audio_encoder, audio_dim, output_dim, task_type, probe_type, dropout, loss_fn):
        super(CLSLayer, self).__init__()
        self.audio_encoder = audio_encoder
        self.audio_dim = audio_dim
        self.probe_type = probe_type
        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout)
        self.loss_fn = loss_fn
        self.to_latent = nn.Identity()

        if task_type == "multilabel":
            self.activation = nn.Sigmoid() # for BCELoss
        elif task_type == "multiclass":
            self.activation = nn.Identity() # CE loss has softmax
        elif task_type == "regression":
            self.activation = nn.Identity() 
        else:
            raise ValueError(f"Unknown task_type {task_type}")

        self.cls_head = nn.Sequential(nn.LayerNorm(audio_dim), nn.Linear(audio_dim, output_dim))

    def forward(self, x, y):
        h_audio = self.audio_encoder(x)
        h_audio = self.to_latent(h_audio[:,0,:])
        h_audio = self.dropout(h_audio)
        z_audio = self.cls_head(h_audio)
        logit = self.activation(z_audio)
        loss = self.loss_fn(logit, y)
        return loss
    
    def test_forward(self, x):
        h_audio = self.audio_encoder(x.squeeze(0))
        h_audio = self.to_latent(h_audio[:,0,:])
        z_audio = self.cls_head(h_audio)
        logit = self.activation(z_audio)
        return logit