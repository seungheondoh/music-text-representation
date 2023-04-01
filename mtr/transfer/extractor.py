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
from mtr.transfer.dataset_wavs.data_manger import get_dataloader
from mtr.utils.train_utils import Logger, AverageMeter, ProgressMeter, EarlyStopping, save_hparams
from mtr.utils.eval_utils import single_query_evaluation, _text_representation
from mtr.utils.transfer_utils import get_model, get_evaluation
from sklearn import metrics
import torch.backends.cudnn as cudnn
from tqdm import tqdm
random.seed(42)
torch.manual_seed(42)
cudnn.deterministic = True

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
parser.add_argument("--eval_dataset", default="fma", type=str)
parser.add_argument("--probe_type", default="extract", type=str)
args = parser.parse_args()


def main(args) -> None:
    save_dir = f"{args.msu_dir}/{args.eval_dataset}/pretrained/{args.framework}_{args.text_type}_{args.text_rep}"
    os.makedirs(save_dir, exist_ok=True)

    model, tokenizer, _ = get_model(args)
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    cudnn.benchmark = True
    model.eval()
    all_loader = get_dataloader(args=args, split="ALL")
    tag_embs, audio_embs = {}, {}
    for batch in tqdm(all_loader):
        audio = batch['audio']
        track_id = str(batch['track_id'][0])
        if args.gpu is not None:
            audio = audio.cuda(args.gpu, non_blocking=True)
        with torch.no_grad():
            z_audio = model.encode_audio(audio.squeeze(0))
        audio_embs[track_id] = z_audio.mean(0).detach().cpu()
    torch.save(audio_embs, os.path.join(save_dir, "audio_embs.pt"))
    
    # text embs
    if args.framework in ['triplet', 'contrastive']:
        for tag in all_loader.dataset.list_of_label:
            input_text = _text_representation(args, tag, tokenizer)
            if args.gpu is not None:
                input_text = input_text.cuda(args.gpu, non_blocking=True)
            with torch.no_grad():
                if args.text_type == "bert":
                    z_tag = model.encode_bert_text(input_text, None)
                elif args.text_type == "glove":
                    z_tag = model.encode_glove_text(input_text)
            tag_embs[tag] = z_tag.detach().cpu()
        torch.save(tag_embs, os.path.join(save_dir, "tag_embs.pt"))

if __name__ == "__main__":
    main(args)