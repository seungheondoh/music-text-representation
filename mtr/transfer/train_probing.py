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
from mtr.utils.transfer_utils import get_cls_config, print_model_params
from sklearn import metrics

parser = argparse.ArgumentParser(description='PyTorch MSD Training')
parser.add_argument('--data_dir', type=str, default="../../../msd-splits/dataset")
parser.add_argument('--msu_dir', type=str, default="../../../MSU/dataset")
parser.add_argument('--framework', type=str, default="contrastive") # or transcription
parser.add_argument('--arch', default='transformer')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--warmup_epochs', default=0, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--min_lr', default=1e-9, type=float)
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=1, type=int,
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
    train_loader = get_dataloader(args=args, audio_embs=audio_embs, split="TRAIN")
    val_loader = get_dataloader(args=args, audio_embs=audio_embs, split="VALID")
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    print_model_params(args, model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_weight_decay) # only update cls
    earlystopping_callback = EarlyStopping(tolerance=20)
    logger = Logger(save_dir)
    save_hparams(args, save_dir)

    best_val_loss = np.inf
    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, model, optimizer, epoch, logger, args)
        val_loss = validate(val_loader, model, epoch, args)
        logger.log_val_loss(val_loss, epoch)
        if val_loss < best_val_loss:
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}, f'{save_dir}/best.pth')
            best_val_loss = val_loss

        earlystopping_callback(val_loss, best_val_loss)
        if earlystopping_callback.early_stop:
            print("We are at epoch:", epoch)
            break

def train(train_loader, model, optimizer, epoch, logger, args):
    train_losses = AverageMeter('Train Loss', ':.4e')
    progress = ProgressMeter(len(train_loader),[train_losses],prefix="Epoch: [{}]".format(epoch))
    iters_per_epoch = len(train_loader)
    model.train()
    for data_iter_step, batch in enumerate(train_loader):
        lr = adjust_learning_rate(optimizer, data_iter_step / iters_per_epoch + epoch, args)
        x = batch['audio']
        y = batch['binary']
        if args.gpu is not None:
            x = x.cuda(args.gpu, non_blocking=True)
            y = y.cuda(args.gpu, non_blocking=True)
        # compute output
        loss = model(x, y)
        train_losses.step(loss.item(), x.size(0))
        logger.log_train_loss(loss, epoch * iters_per_epoch + data_iter_step)
        logger.log_learning_rate(lr, epoch * iters_per_epoch + data_iter_step)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if data_iter_step % args.print_freq == 0:
            progress.display(data_iter_step)

def validate(val_loader, model, epoch, args):
    losses_val = AverageMeter('Valid Loss', ':.4e')
    progress_val = ProgressMeter(len(val_loader),[losses_val],prefix="Epoch: [{}]".format(epoch))
    model.eval()
    epoch_end_loss = []
    for data_iter_step, batch in enumerate(val_loader):
        x = batch['audio']
        y = batch['binary']
        if args.gpu is not None:
            x = x.cuda(args.gpu, non_blocking=True)
            y = y.cuda(args.gpu, non_blocking=True)
        with torch.no_grad():
            loss = model(x, y) # flatten batch
        epoch_end_loss.append(loss.detach().cpu())
        losses_val.step(loss.item(), x.size(0))
        if data_iter_step % args.print_freq == 0:
            progress_val.display(data_iter_step)
    val_loss = torch.stack(epoch_end_loss).mean(0, False)
    return val_loss
    
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


if __name__ == '__main__':
    main()

    