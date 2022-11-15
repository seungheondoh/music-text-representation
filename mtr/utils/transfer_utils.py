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
from sklearn import metrics


def compute_accuracy_metrics(ground_truth, predicted, threshold=0.5):
    decisions = predicted > threshold
    binary_pred = decisions.astype(np.int16)
    return metrics.classification_report(ground_truth, binary_pred, output_dict=True)

def get_binary_decisions(ground_truth, predicted):
    """https://github.com/MTG/mtg-jamendo-dataset/blob/31507d6e9a64da11471bb12168372db4f98d7783/scripts/mediaeval/calculate_decisions.py#L8"""
    thresholds = {}
    avg_fscore_macro, avg_fscore_weighted = [], []
    for idx in range(len(ground_truth[0])):
        precision, recall, threshold = metrics.precision_recall_curve(
            ground_truth[:, idx], predicted[:, idx])
        f_score = np.nan_to_num(
            (2 * precision * recall) / (precision + recall))
        thresholds[idx] = threshold[np.argmax(f_score)]

        results = compute_accuracy_metrics(ground_truth[:, idx], predicted[:, idx], threshold=0.5)
        avg_fscore_weighted.append(results['weighted avg']['f1-score'])
        avg_fscore_macro.append(results['macro avg']['f1-score'])
    avg_macro_f1 = np.array(avg_fscore_macro).mean()
    avg_fscore_weighted = np.array(avg_fscore_weighted).mean()
    bestF1_decisions = predicted > np.array(list(thresholds.values()))
    return thresholds, avg_macro_f1, avg_fscore_weighted, bestF1_decisions

def get_evaluation(binary, logit, labels, task_type):
    if task_type == "multilabel":
        _, avg_macro_f1, avg_fscore_weighted, bestF1_decisions = get_binary_decisions(binary, logit)
        roc_auc = metrics.roc_auc_score(binary, logit, average='macro')
        pr_auc = metrics.average_precision_score(binary, logit, average='macro')
        results = {
            "macro_roc": roc_auc,
            "macro_pr": pr_auc,
            "macro_fl": metrics.f1_score(binary, bestF1_decisions, average='macro'),
            "avg_macro_f1": avg_macro_f1,
            "avg_fscore_weighted": avg_fscore_weighted
        }
        # tag wise score
        roc_aucs = metrics.roc_auc_score(binary, logit, average=None)
        pr_aucs = metrics.average_precision_score(binary, logit, average=None)
        tag_wise = {}
        for i in range(len(labels)):
            tag_wise[labels[i]] = {
                "roc_auc":roc_aucs[i], 
                "pr_auc":pr_aucs[i]
        }
        results['tag_wise'] = tag_wise
    else:
        results = {
            "acc": metrics.accuracy_score(binary.argmax(axis=1), logit.argmax(axis=1))
        }
    return results

def get_model(args):
    save_dir = f"../{args.framework}/exp/{args.arch}_{args.frontend}_{args.mix_type}_{args.audio_rep}/{args.text_type}_{args.text_rep}"
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
    if args.framework == "contrastive":
        model = ContrastiveModel(
            audio_encoder= audio_encoder,
            text_encoder= text_encoder,
            text_type = config.text_type,
            audio_dim= config.audio_dim,
            text_dim= config.text_dim,
            mlp_dim= config.mlp_dim,
            temperature = config.temperature
        )
    elif args.framework == "triplet":
        model = TripletModel(
            audio_encoder= audio_encoder,
            text_encoder= text_encoder,
            text_type = config.text_type,
            audio_dim= config.audio_dim,
            text_dim= config.text_dim,
            mlp_dim= config.mlp_dim,
            margin = config.margin
        )
    elif args.framework == "classification":
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
    return model, tokenizer, config # zeroshot need text and audio encoder
        
def get_cls_config(args):
    task_type = "multilabel"
    loss_fn = nn.BCELoss()
    print(args.eval_dataset)
    if args.eval_dataset in ["msd", "mtat", 'mtg_top50tags']:
        output_dim = 50
    elif args.eval_dataset == "kvt":
        output_dim = 42
    elif args.eval_dataset == "mtg_genre":
        output_dim = 87
    elif args.eval_dataset == "mtg_instrument":
        output_dim = 40
    elif args.eval_dataset == "mtg_moodtheme":
        output_dim = 56
    elif args.eval_dataset == "openmic":
        output_dim = 20
    elif args.eval_dataset == "gtzan":
        task_type = "multiclass"
        output_dim = 10
        loss_fn = nn.CrossEntropyLoss()
    elif args.eval_dataset == "emotify":
        task_type = "multiclass"
        output_dim = 9
        loss_fn = nn.CrossEntropyLoss()
    elif args.eval_dataset == "fma":
        task_type = "multiclass"
        output_dim = 8
        loss_fn = nn.CrossEntropyLoss()
    return task_type, output_dim, loss_fn


def print_model_params(args, model):
    n_parameters = sum(p.numel() for p in model.parameters())
    train_n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("============")
    print("lr: %.2e" % (args.lr))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))
    print('number train of params (M): %.2f' % (train_n_parameters / 1.e6))
    print("============")