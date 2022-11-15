import torch
import torch.nn as nn

class ProbingLayer(nn.Module):
    def __init__(self, audio_dim, mlp_dim, output_dim, task_type, probe_type, dropout, is_norm, loss_fn):
        super(ProbingLayer, self).__init__()
        self.audio_dim = audio_dim
        self.task_type = task_type
        self.probe_type = probe_type
        self.mlp_dim = mlp_dim
        self.output_dim = output_dim
        self.is_norm = is_norm
        self.dropout = nn.Dropout(dropout)
        self.loss_fn = loss_fn
        if self.is_norm:
            self.norm_layer = nn.LayerNorm(audio_dim)
        
        if task_type == "multilabel":
            self.activation = nn.Sigmoid()
        elif task_type == "multiclass":
            self.activation = nn.Identity() # CE loss
        elif task_type == "regression":
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unknown task_type {task_type}")

        if self.probe_type == "mlp":
            self.cls_head = nn.Sequential(
                nn.Linear(audio_dim, self.mlp_dim),
                nn.ReLU(),
                nn.Linear(self.mlp_dim, output_dim),
            )
        else:
            self.cls_head = nn.Linear(audio_dim, output_dim)

    def forward(self, x, y):
        if self.is_norm:
            x = self.norm_layer(x)
        x = self.dropout(x)
        output = self.cls_head(x)
        logit = self.activation(output)
        if self.task_type == "multiclass":
            loss = self.loss_fn(logit, y)
        else:
            loss = self.loss_fn(logit, y)
        return loss
    
    def test_forward(self, x):
        output = self.cls_head(x)
        logit = self.activation(output)
        return logit