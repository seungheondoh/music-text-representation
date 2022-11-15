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
import pandas as pd
import json

from mtr.classification.dataset import ECALS_Dataset
from mtr.modules.audio_rep import TFRep
from mtr.modules.tokenizer import ResFrontEnd, SpecPatchEmbed
from mtr.modules.encoder import MusicTransformer
from mtr.classification.model import ClassificationModel
from mtr.utils.transfer_utils import get_binary_decisions
from mtr.utils.eval_utils import rank_eval

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
parser.add_argument('--framework', type=str, default="tagging") # or transcription
parser.add_argument('--data_dir', type=str, default="../../../msd-splits/dataset")
parser.add_argument('--arch', default='transformer')
parser.add_argument('-j', '--workers', default=24, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--warmup_epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
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
parser.add_argument('--gpu', default=0, type=int,
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
parser.add_argument("--text_type", default="tag", type=str)
parser.add_argument("--text_rep", default="binary", type=str)

def main():
    args = parser.parse_args()
    main_worker(args)

def main_worker(args):
    test_dataset = ECALS_Dataset(
        args.data_dir, "TEST", args.sr, args.duration, args.num_chunks, False
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=None,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False)
    

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

    model = ClassificationModel(
        audio_encoder=audio_encoder,
        audio_dim=args.attention_ndim,
        mlp_dim=args.mlp_dim, 
        num_classes = len(test_dataset.list_of_label)
    )
    save_dir = f"exp/{args.arch}_{args.frontend}_{args.mix_type}_{args.audio_rep}/{args.text_rep}_{args.text_type}/"
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

    model.eval()
    audio_dict, predictions, groudturths, track_ids = {}, [], [], []
    for batch in tqdm(test_loader):
        x = batch['audio']
        y = batch['binary']
        track_id = batch['track_id']
        if args.gpu is not None:
            x = x.cuda(args.gpu, non_blocking=True)
            y = y.cuda(args.gpu, non_blocking=True)
        with torch.no_grad():
            z_audio = model.encode_audio(x.squeeze(0)) # flatten batch
            predict = model.forward_eval(x.squeeze(0)) # flatten batch
        audio_dict[track_id[0]] = z_audio.mean(0).detach().cpu()
        predictions.append(predict.mean(0,True).detach().cpu())
        groudturths.append(y.detach().cpu())
        track_ids.append(track_id)
    
    tag_dict = {}
    for tag, centorid in zip(test_dataset.list_of_label, model.head.fc_cls.weight):
        tag_dict[tag] = centorid.detach().cpu()
    torch.save(audio_dict, os.path.join(save_dir, "audio_embs.pt"))
    torch.save(tag_dict, os.path.join(save_dir, "tag_embs.pt"))

    logits = torch.cat(predictions, dim=0).numpy()
    targets = torch.cat(groudturths, dim=0).numpy()
    _, _, _, bestF1_decisions = get_binary_decisions(targets, logits)
    multi_query_gt = json.load(open(os.path.join(args.data_dir, "ecals_annotation/multiquery_samples.json"), 'r'))
    gt_items, pred_items = multi_query_annotation(bestF1_decisions,track_ids,test_dataset.list_of_label,multi_query_gt)

    logits = pd.DataFrame(logits, columns=test_dataset.list_of_label)
    targets = pd.DataFrame(targets, columns=test_dataset.list_of_label)
    single_query_evaluation(targets, logits, save_dir, test_dataset.list_of_label)
    single_query_evaluation(targets, logits, save_dir, TAGNAMES)
    # multiquery_annotation
    mq_results = rank_eval(gt_items, pred_items) # multi_query evaluation
    with open(os.path.join(save_dir, f"mq_results.json"), mode="w") as io:
        json.dump(mq_results, io, indent=4)


def single_query_evaluation(targets, logits, save_dir, labels):
    targets = targets[labels]
    logits = logits[labels]
    roc_auc = metrics.roc_auc_score(targets, logits, average='macro')
    pr_auc = metrics.average_precision_score(targets, logits, average='macro')
    results = {
        'roc_auc' :roc_auc,
        'pr_auc': pr_auc
    }
    # tag wise score
    roc_aucs = metrics.roc_auc_score(targets, logits, average=None)
    pr_aucs = metrics.average_precision_score(targets, logits, average=None)
    tag_wise = {}
    for i in range(len(labels)):
        tag_wise[labels[i]] = {
            "roc_auc":roc_aucs[i], 
            "pr_auc":pr_aucs[i]
    }
    results['tag_wise'] = tag_wise
    label_len = len(labels)
    with open(os.path.join(save_dir, f"{label_len}_results.json"), mode="w") as io:
        json.dump(results, io, indent=4)

def tag_to_binary(tag_list, list_of_label, tag_to_idx):
    bainry = np.zeros([len(list_of_label),], dtype=np.float32)
    for tag in tag_list:
        bainry[tag_to_idx[tag]] = 1.0
    return bainry

def multi_query_annotation(bestF1_decisions,track_ids,list_of_label,multi_query_gt):
    tag_to_idx = {i:idx for idx, i in enumerate(list_of_label)}
    msd_ids = [i[0] for i in track_ids]
    df_annotation = pd.DataFrame(bestF1_decisions, index=msd_ids, columns=list_of_label)
    predict_multi_query, predict_binary = {}, {}
    for idx in range(len(df_annotation)):
        item = df_annotation.iloc[idx]
        track_id = item.name
        if track_id in multi_query_gt.keys():
            tagging_list = list(item[item == True].index)
            predict_multi_query[track_id] = tagging_list
            predict_binary[track_id] = tag_to_binary(tagging_list, list_of_label, tag_to_idx)

    gt_items = {v:k for k,v in multi_query_gt.items()}
    gt_cap_to_binary = {}
    for k,v in multi_query_gt.items():
        tag_list = [tag.strip() for tag in v.split(", ")]
        gt_cap_to_binary[v] = tag_to_binary(tag_list, list_of_label, tag_to_idx)

    captions = [k for k in gt_cap_to_binary.keys()]
    gt_binarys = [gt_cap_to_binary[cap] for cap in captions]
    pre_ids = [k for k in predict_binary.keys()]
    pred_binarys = [predict_binary[ids] for ids in pre_ids]
    gts = torch.from_numpy(np.stack(gt_binarys))
    preds = torch.from_numpy(np.stack(pred_binarys))
    tag_matching = preds @ gts.T # measure tag matching
    df_pred = pd.DataFrame(tag_matching.numpy(), index=pre_ids, columns=captions).T
    pred_items = {}
    for idx in range(len(df_pred)):
        item = df_pred.iloc[idx]
        pred_items[item.name] = list(item.sort_values(ascending=False).index)
    return gt_items, pred_items
            

if __name__ == '__main__':
    main()

    