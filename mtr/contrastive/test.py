import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from sklearn import metrics
from tqdm import tqdm
import json
import pandas as pd
from transformers import AutoModel, AutoTokenizer, set_seed
from mtr.contrastive.dataset import ECALS_Dataset
from mtr.modules.audio_rep import TFRep
from mtr.modules.tokenizer import ResFrontEnd, SpecPatchEmbed
from mtr.modules.encoder import MusicTransformer
from mtr.contrastive.model import ContrastiveModel
from mtr.utils.eval_utils import single_query_evaluation, multi_query_evaluation, _text_representation

TAGNAMES = [
    'rock','pop','indie','alternative','electronic','hip hop','metal','jazz','punk',
    'folk','alternative rock','indie rock','dance','hard rock','00s','soul','hardcore',
    '80s','country','classic rock','punk rock','blues','chillout','experimental',
    'heavy metal','death metal','90s','reggae','progressive rock','ambient','acoustic',
    'beautiful','british','rnb','funk','metalcore','mellow','world','guitar','trance',
    'indie pop','christian','house','spanish','latin','psychedelic','electro','piano',
    '70s','progressive metal',
]

parser = argparse.ArgumentParser(description='PyTorch MSD Training')
parser.add_argument('--framework', type=str, default="contrastive") # or transcription
parser.add_argument("--text_backbone", default="bert-base-uncased", type=str)
parser.add_argument('--data_dir', type=str, default="../../../msd-splits/dataset")
parser.add_argument('--arch', default='transformer')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--min_lr', default=1e-9, type=float)
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:12312', type=str,
                    help='url used to set up distributed training') # env://
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=1, type=int,
                    help='GPU id to use.')
parser.add_argument('--print_freq', default=50, type=int)
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
# train detail
parser.add_argument("--duration", default=9.91, type=int)
parser.add_argument("--sr", default=16000, type=int)
parser.add_argument("--num_chunks", default=3, type=int)
parser.add_argument("--mel_dim", default=128, type=int)
parser.add_argument("--n_fft", default=1024, type=int)
parser.add_argument("--win_length", default=1024, type=int)
parser.add_argument("--frontend", default="cnn", type=str)
parser.add_argument("--mix_type", default="cf", type=str)
parser.add_argument("--audio_rep", default="mel", type=str)
parser.add_argument("--cos", default=True, type=bool)
parser.add_argument("--attention_nlayers", default=4, type=int)
parser.add_argument("--attention_ndim", default=256, type=int)
parser.add_argument("--temperature", default=0.2, type=float)
parser.add_argument("--mlp_dim", default=128, type=int)
parser.add_argument("--text_type", default="bert", type=str)
parser.add_argument("--text_rep", default="stocastic", type=str)

def main():
    args = parser.parse_args()
    main_worker(args)

def main_worker(args):
    audio_preprocessr = TFRep(
                sample_rate= args.sr,
                f_min=0,
                f_max= int(args.sr / 2),
                n_fft = args.n_fft,
                win_length = args.win_length,
                hop_length = int(0.01 * args.sr),
                n_mels = args.mel_dim
    )

    frontend = ResFrontEnd(
        input_size=(args.mel_dim, int(100 * args.duration) + 1), # 128 * 992
        conv_ndim=128, 
        attention_ndim=args.attention_ndim,
        mix_type= args.mix_type
    )
    audio_encoder = MusicTransformer(
        audio_representation=audio_preprocessr,
        frontend = frontend,
        audio_rep = args.audio_rep,
        attention_nlayers= args.attention_nlayers,
        attention_ndim= args.attention_ndim
    )
    if args.text_type == "bert":
        text_encoder = AutoModel.from_pretrained(args.text_backbone)
        tokenizer = AutoTokenizer.from_pretrained(args.text_backbone)
        args.text_dim = 768
    elif args.text_type == "glove":
        text_encoder = nn.Identity()
        tokenizer = torch.load(os.path.join(args.data_dir, "ecals_annotation", "glove_tag_embs.pt"))
        args.text_dim = 300

    args.audio_dim = args.attention_ndim
    model = ContrastiveModel(
        audio_encoder= audio_encoder,
        text_encoder= text_encoder,
        text_type = args.text_type,
        audio_dim= args.audio_dim,
        text_dim= args.text_dim,
        mlp_dim= args.mlp_dim,
        temperature = args.temperature
    )
    save_dir = f"exp/{args.arch}_{args.frontend}_{args.mix_type}_{args.audio_rep}/{args.text_type}_{args.text_rep}"
    pretrained_object = torch.load(f'{save_dir}/best.pth', map_location='cpu')
    state_dict = pretrained_object['state_dict']
    for k in list(state_dict.keys()):
        if k.startswith('module.'):
            state_dict[k[len("module."):]] = state_dict[k]
        del state_dict[k]
    model.load_state_dict(state_dict)


    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    cudnn.benchmark = True

    test_dataset= ECALS_Dataset(
        data_path = args.data_dir,
        split = "TEST",
        sr = args.sr,
        duration = args.duration,
        num_chunks = args.num_chunks,
        text_preprocessor= tokenizer,
        text_type=args.text_type,
        text_rep = "caption"
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=None,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False)

    multi_query_set = json.load(open(os.path.join(args.data_dir, "ecals_annotation", "multiquery_samples.json"),'r'))
    model.eval()
    track_ids, audio_embs, groudturths, audio_dict, multi_query_dict = [], [], [], {}, {}
    for batch in tqdm(test_loader):
        audio = batch['audio']
        list_of_tag = [tag[0] for tag in batch['text']]
        caption = ", ".join(list_of_tag)
        track_id = batch['track_id'][0]
        track_ids.append(track_id)
        groudturths.append(batch['binary'])
        input_text = _text_representation(args, list_of_tag, tokenizer)
        if args.gpu is not None:
            audio = audio.cuda(args.gpu, non_blocking=True)
            input_text = input_text.cuda(args.gpu, non_blocking=True)
        with torch.no_grad():
            z_audio = model.encode_audio(audio.squeeze(0))
            if args.text_type == "bert":
                z_caption = model.encode_bert_text(input_text, None)
            elif args.text_type == "glove":
                z_caption = model.encode_glove_text(input_text)
        if track_id in multi_query_set.keys():
            multi_query_dict[track_id] = {
                "z_audio": z_audio.mean(0).detach().cpu(),
                "track_id": track_id,
                "text": caption,
                "binary": batch['binary'],
                "z_text": z_caption.detach().cpu()
            }
        audio_embs.append(z_audio.mean(0).detach().cpu())
        audio_dict[track_id] = z_audio.mean(0).detach().cpu()
    audio_embs = torch.stack(audio_embs, dim=0)
    # single query evaluation
    tag_embs, tag_dict = [], {}
    for tag in test_dataset.list_of_label:
        input_text = _text_representation(args, tag, tokenizer)
        if args.gpu is not None:
            input_text = input_text.cuda(args.gpu, non_blocking=True)
        with torch.no_grad():
            if args.text_type == "bert":
                z_tag = model.encode_bert_text(input_text, None)
            elif args.text_type == "glove":
                z_tag = model.encode_glove_text(input_text)
        tag_embs.append(z_tag.detach().cpu())
        tag_dict[tag] = z_tag.detach().cpu()

    torch.save(audio_dict, os.path.join(save_dir, "audio_embs.pt"))
    torch.save(tag_dict, os.path.join(save_dir, "tag_embs.pt"))
    torch.save(multi_query_dict, os.path.join(save_dir, "caption_embs.pt"))

    tag_embs = torch.cat(tag_embs, dim=0)
    targets = torch.cat(groudturths, dim=0)
    audio_embs = nn.functional.normalize(audio_embs, dim=1)
    tag_embs = nn.functional.normalize(tag_embs, dim=1)
    single_query_logits = audio_embs @ tag_embs.T

    sq_logits = pd.DataFrame(single_query_logits.numpy(), index=track_ids, columns=test_dataset.list_of_label)
    sq_targets = pd.DataFrame(targets.numpy(), index=track_ids, columns=test_dataset.list_of_label)

    single_query_evaluation(sq_targets, sq_logits, save_dir, TAGNAMES) # 50 tag evaluation
    single_query_evaluation(sq_targets, sq_logits, save_dir, test_dataset.list_of_label) # 1054 tag evaluation
    multi_query_evaluation(tag_dict, multi_query_dict, save_dir) # multi_query evaluation

if __name__ == '__main__':
    main()