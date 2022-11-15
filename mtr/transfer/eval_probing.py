import os
import json
import math
import argparse
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

import torch
import torch.backends.cudnn as cudnn
# backbones
from mtr.transfer.model_probing import ProbingLayer
from mtr.transfer.dataset_embs.data_manger import get_dataloader
from mtr.utils.train_utils import Logger, AverageMeter, ProgressMeter, EarlyStopping, save_hparams
from mtr.utils.eval_utils import single_query_evaluation, _text_representation
from mtr.utils.transfer_utils import get_cls_config, get_evaluation
from sklearn import metrics

parser = argparse.ArgumentParser(description='PyTorch MSD Training')
parser.add_argument('--data_dir', type=str, default="../../../msd-splits/dataset")
parser.add_argument('--msu_dir', type=str, default="../../../MSU/dataset")
parser.add_argument('--framework', type=str, default="contrastive") # or transcription
parser.add_argument('--arch', default='transformer')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--warmup_epochs', default=0, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--min_lr', default=1e-9, type=float)
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--print_freq', default=100, type=int)
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
parser.add_argument("--probe_type", default="mlp", type=str)
parser.add_argument("--mlp_dim", default=512, type=int)
parser.add_argument("--eval_dataset", default="gtzan", type=str)
parser.add_argument("--dropout", default=0, type=float)
parser.add_argument("--is_norm", default=0, type=int)
parser.add_argument("--l2_weight_decay", default=0, type=float)

args = parser.parse_args()

def main():
    save_dir = f"exp/{args.probe_type}/{args.eval_dataset}/{args.framework}_{args.text_type}_{args.text_rep}/{args.lr}_{args.batch_size}_{args.dropout}_{args.is_norm}"
    embs_dir = f"{args.msu_dir}/{args.eval_dataset}/pretrained/{args.framework}_{args.text_type}_{args.text_rep}"
    if args.eval_dataset in ["mtg_top50tags", "mtg_genre", "mtg_instrument", "mtg_moodtheme"]:
        embs_dir = f"{args.msu_dir}/mtg/pretrained/{args.framework}_{args.text_type}_{args.text_rep}"
    audio_embs = torch.load(os.path.join(embs_dir, 'audio_embs.pt'))
    audio_dim = 128

    task_type, output_dim, loss_fn = get_cls_config(args)
    model = ProbingLayer(
        audio_dim = audio_dim,
        mlp_dim = args.mlp_dim,
        output_dim = output_dim,
        task_type = task_type,
        probe_type = args.probe_type,
        dropout = args.dropout,
        is_norm = args.is_norm,
        loss_fn = loss_fn
    )
    pretrained_object = torch.load(f'{save_dir}/best.pth', map_location='cpu')
    state_dict = pretrained_object['state_dict']
    model.load_state_dict(state_dict)

    test_loader = get_dataloader(args=args, audio_embs=audio_embs, split="TEST")
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    cudnn.benchmark = True
    model.eval()
    predictions, groudturths = [], []
    for batch in tqdm(test_loader):
        x = batch['audio']
        y = batch['binary']
        if args.gpu is not None:
            x = x.cuda(args.gpu, non_blocking=True)
            y = y.cuda(args.gpu, non_blocking=True)
        with torch.no_grad():
            predict = model.test_forward(x) # flatten batch
        predictions.append(predict.mean(0, True).detach().cpu())
        groudturths.append(y.detach().cpu())
    
    logits = torch.cat(predictions, dim=0)
    targets = torch.cat(groudturths, dim=0)
    if args.eval_dataset in ['fma', 'gtzan', 'emotify']:
        results = get_evaluation(targets.numpy(), logits.numpy(), test_loader.dataset.list_of_label, 'multiclass')
    else:
        results = get_evaluation(targets.numpy(), logits.numpy(),test_loader.dataset.list_of_label, 'multilabel')
    with open(os.path.join(save_dir, f"results.json"), mode="w") as io:
        json.dump(results, io, indent=4)

if __name__ == '__main__':
    main()

    