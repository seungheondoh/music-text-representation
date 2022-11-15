
import os
import torch
from torch import nn
import numpy as np
from omegaconf import OmegaConf
from transformers import AutoModel, AutoTokenizer, set_seed
from mtr.modules.audio_rep import TFRep
from mtr.modules.tokenizer import ResFrontEnd, SpecPatchEmbed
from mtr.modules.encoder import MusicTransformer
from mtr.modules.head import ClsHead
from mtr.classification.model import ClassificationModel
from mtr.triplet.model import TripletModel
from mtr.contrastive.model import ContrastiveModel

def get_model(framework, text_type, text_rep, arch='transformer', frontend='cnn', mix_type="cf", audio_rep="mel"):
    save_dir = f"../mtr/{framework}/exp/{arch}_{frontend}_{mix_type}_{audio_rep}/{text_type}_{text_rep}"
    config = OmegaConf.load(os.path.join(save_dir, "hparams.yaml"))
    audio_preprocessr = TFRep(
                sample_rate= config.sr,
                f_min=0,
                f_max= int(config.sr / 2),
                n_fft = config.n_fft,
                win_length = config.win_length,
                hop_length = int(0.01 * config.sr),
                n_mels = config.mel_dim
    )
    frontend = ResFrontEnd(
        input_size=(config.mel_dim, int(100 * config.duration) + 1), # 128 * 992
        conv_ndim=128, 
        attention_ndim=config.attention_ndim,
        mix_type= config.mix_type
    )
    audio_encoder = MusicTransformer(
        audio_representation=audio_preprocessr,
        frontend = frontend,
        audio_rep = config.audio_rep,
        attention_nlayers= config.attention_nlayers,
        attention_ndim= config.attention_ndim
    )
    if config.text_type == "bert":
        text_encoder = AutoModel.from_pretrained(config.text_backbone)
        tokenizer = AutoTokenizer.from_pretrained(config.text_backbone)
        config.text_dim = 768
    elif config.text_type == "glove":
        text_encoder = nn.Identity()
        tokenizer = torch.load(os.path.join(config.data_dir, "ecals_annotation", "glove_tag_embs.pt"))
        add_tokenizer = torch.load(os.path.join(args.msu_dir, "pretrain", "glove_tag_embs.pt"))
        tokenizer.update(add_tokenizer)
        config.text_dim = 300
    else:
        tokenizer = None
    config.audio_dim = config.attention_ndim
    if framework == "contrastive":
        model = ContrastiveModel(
            audio_encoder= audio_encoder,
            text_encoder= text_encoder,
            text_type = config.text_type,
            audio_dim= config.audio_dim,
            text_dim= config.text_dim,
            mlp_dim= config.mlp_dim,
            temperature = config.temperature
        )
    elif framework == "triplet":
        model = TripletModel(
            audio_encoder= audio_encoder,
            text_encoder= text_encoder,
            text_type = config.text_type,
            audio_dim= config.audio_dim,
            text_dim= config.text_dim,
            mlp_dim= config.mlp_dim,
            margin = config.margin
        )
    elif framework == "classification":
        model = ClassificationModel(
        audio_encoder=audio_encoder,
        audio_dim=config.attention_ndim,
        mlp_dim=config.mlp_dim, 
        num_classes = 1054
    )
    pretrained_object = torch.load(f'{save_dir}/best.pth', map_location='cpu')
    state_dict = pretrained_object['state_dict']
    for k in list(state_dict.keys()):
        if k.startswith('module.'):
            state_dict[k[len("module."):]] = state_dict[k]
        del state_dict[k]              
    model.load_state_dict(state_dict) 
    return model, tokenizer, config