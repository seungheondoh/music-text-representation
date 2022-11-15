import json
import os
import argparse
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import torch.backends.cudnn as cudnn
# backbones
from mtr.transfer.dataset_embs.data_manger import get_dataloader
from mtr.utils.train_utils import Logger, AverageMeter, ProgressMeter, EarlyStopping, save_hparams
from mtr.utils.eval_utils import single_query_evaluation, _text_representation
from mtr.utils.transfer_utils import get_model, get_evaluation
from sklearn import metrics

parser = argparse.ArgumentParser(description='PyTorch MSD Training')
parser.add_argument('--data_dir', type=str, default="../../../msd-splits/dataset")
parser.add_argument('--msu_dir', type=str, default="../../../MSU/dataset")
parser.add_argument('--framework', type=str, default="contrastive") # or transcription
parser.add_argument('--arch', default='transformer')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--warmup_epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--min_lr', default=1e-9, type=float)
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
# train detail
parser.add_argument("--duration", default=9.91, type=int)
parser.add_argument("--sr", default=16000, type=int)
parser.add_argument("--num_chunks", default=8, type=int)
parser.add_argument("--mel_dim", default=128, type=int)
parser.add_argument("--n_fft", default=1024, type=int)
parser.add_argument("--win_length", default=1024, type=int)
parser.add_argument("--frontend", default="cnn", type=str)
parser.add_argument("--mix_type", default="cf", type=str)
parser.add_argument("--audio_rep", default="mel", type=str)
parser.add_argument("--text_type", default="bert", type=str)
parser.add_argument("--text_rep", default="tag", type=str)
parser.add_argument("--cos", default=True, type=bool)
parser.add_argument("--attention_nlayers", default=4, type=int)
parser.add_argument("--attention_ndim", default=256, type=int)
parser.add_argument("--text_backbone", default="bert-base-uncased", type=str)
# downstream options
parser.add_argument("--probe_type", default="zs", type=str)
parser.add_argument("--eval_dataset", default="fma", type=str)
args = parser.parse_args()

def main(args) -> None:
    save_dir = f"exp/{args.probe_type}/{args.eval_dataset}/{args.framework}_{args.text_type}_{args.text_rep}/"
    save_hparams(args, save_dir)
    embs_dir = f"{args.msu_dir}/{args.eval_dataset}/pretrained/{args.framework}_{args.text_type}_{args.text_rep}"
    if args.eval_dataset in ["mtg_top50tags", "mtg_genre", "mtg_instrument", "mtg_moodtheme"]:
        embs_dir = f"{args.msu_dir}/mtg/pretrained/{args.framework}_{args.text_type}_{args.text_rep}"
    audio_embs = torch.load(os.path.join(embs_dir, 'audio_embs.pt'))
    tag_embs = torch.load(os.path.join(embs_dir, 'tag_embs.pt'))
    test_loader = get_dataloader(args=args, audio_embs=audio_embs, split="TEST")
    t_embs = [tag_embs[tag] for tag in test_loader.dataset.list_of_label]
    a_embs, groudturths = [], []
    for batch in tqdm(test_loader):
        a_embs.append(batch['audio'])
        groudturths.append(batch['binary'])
    a_embs = torch.cat(a_embs, dim=0)
    t_embs = torch.cat(t_embs, dim=0)
    targets = torch.cat(groudturths, dim=0)

    a_embs = nn.functional.normalize(a_embs, dim=1)
    t_embs = nn.functional.normalize(t_embs, dim=1)
    logits = a_embs @ t_embs.T
    if args.eval_dataset in ['fma', 'gtzan', 'emotify']:
        results = get_evaluation(targets.numpy(),logits.numpy(), test_loader.dataset.list_of_label, 'multiclass')
    else:
        results = get_evaluation(targets.numpy(),logits.numpy(),test_loader.dataset.list_of_label, 'multilabel')
    with open(os.path.join(save_dir, f"results.json"), mode="w") as io:
        json.dump(results, io, indent=4)
    
if __name__ == "__main__":
    main(args)