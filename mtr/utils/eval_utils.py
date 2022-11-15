import os
import json
import torch
import numpy as np
import pandas as pd
from sklearn import metrics
from astropy.stats import jackknife

def _text_representation(args, text, tokenizer):
    if args.text_type == "bert":
        if isinstance(text, str):
            text_inputs = tokenizer(text, return_tensors="pt")['input_ids']
        elif isinstance(text, list):
            text_inputs = tokenizer(", ".join(text), return_tensors="pt")['input_ids']
    elif args.text_type == "glove":
        if isinstance(text, str):
            text_inputs = torch.from_numpy(np.array(tokenizer[text]).astype("float32")).unsqueeze(0)
        elif isinstance(text, list):
            tag_embs= np.array([tokenizer[tag] for tag in text]).astype("float32")
            text_inputs = torch.from_numpy(tag_embs.mean(axis=0)).unsqueeze(0)
    else:
        print("error!")
    return text_inputs 
    
def single_query_evaluation(targets, logits, save_dir, labels):
    """
    target = Pandas DataFrame Binary Mtrix( track x label )
    logits = Pandas DataFrame Logit Mtrix( track x label )
    label = tag list
    """
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
    
def multi_query_evaluation(tag_dict, multi_query_dict, save_dir):
    track_ids = [k for k in multi_query_dict.keys()]
    tags = [tag for tag in tag_dict.keys()]
    tag_embs = torch.cat([tag_dict[tag] for tag in tags], dim=0)
    audio_embs = torch.stack([multi_query_dict[k]['z_audio'] for k in track_ids])
    text_embs = torch.cat([multi_query_dict[k]['z_text'] for k in track_ids], dim=0)
    gt_items = {", ".join(multi_query_dict[k]['text']):k for k in track_ids} # unique caption case
    
    tag_embs = torch.nn.functional.normalize(tag_embs, dim=1)
    audio_embs = torch.nn.functional.normalize(audio_embs, dim=1)
    text_embs = torch.nn.functional.normalize(text_embs, dim=1)

    logits = text_embs @ audio_embs.T # text to audio
    sq_logits = tag_embs @ tag_embs.T # tag similarity/
    mq_logits = text_embs @ text_embs.T # caption similarity
    df_pred = pd.DataFrame(logits.numpy(), index=gt_items.keys(), columns=gt_items.values())
    pred_items = {}
    for idx in range(len(df_pred)):
        item = df_pred.iloc[idx]
        pred_items[item.name] = list(item.sort_values(ascending=False).index)
    results = rank_eval(gt_items, pred_items)
    print(results)
    results['tag_stat'] = {
        "mean": float(sq_logits.mean()),
        "std": float(sq_logits.std())
    }
    results['caption_stat'] = {
        "mean": float(mq_logits.mean()),
        "std": float(mq_logits.std())
    }
    results['audio_stat'] = {
        "mean": float(logits.mean()),
        "std": float(logits.std())
    }
    with open(os.path.join(save_dir, f"mq_results.json"), mode="w") as io:
        json.dump(results, io, indent=4)

def rank_eval(gt_items, pred_items):
    """
        gt_items = Dict: caption -> msdid
        pred_items = Dict: caption -> [msdid, msdid, msdid, ...] sort by relevant score
    """
    R1, R5, R10, mAP10, med_rank = [], [], [], [], []
    for i, cap in enumerate(gt_items):
        gt_fname = gt_items[cap]
        pred_fnames = pred_items[cap]
        preds = np.asarray([gt_fname == pred for pred in pred_fnames])
        rank_value = min([idx for idx, retrieval_success in enumerate(preds) if retrieval_success])
        R1.append(np.sum(preds[:1], dtype=float))
        R5.append(np.sum(preds[:5], dtype=float))
        R10.append(np.sum(preds[:10], dtype=float))
        positions = np.arange(1, 11, dtype=float)[preds[:10] > 0]
        med_rank.append(rank_value)
        if len(positions) > 0:
            precisions = np.divide(np.arange(1, len(positions) + 1, dtype=float), positions)
            avg_precision = np.sum(precisions, dtype=float)
            mAP10.append(avg_precision)
        else:
            mAP10.append(0.0)

    r1_estimate, _, _, _ = jackknife.jackknife_stats(np.asarray(R1), np.mean, 0.95)
    r5_estimate, _, _, _ = jackknife.jackknife_stats(np.asarray(R5), np.mean, 0.95)
    r10_estimate, _, _, _ = jackknife.jackknife_stats(np.asarray(R10), np.mean, 0.95)
    map_estimate, _, _, _ = jackknife.jackknife_stats(np.asarray(mAP10), np.mean, 0.95)
    medrank_estimate, _, _, _ = jackknife.jackknife_stats(np.asarray(med_rank), np.median, 0.95)
    return {
        "R@1": r1_estimate,
        "R@5": r5_estimate,
        "R@10": r10_estimate,
        "mAP@10": map_estimate,
        "medRank": medrank_estimate,
    }

def _medrank(logits):
    rank_values = []
    for idx in range(len(logits)):
        _, idx_list = logits[idx].topk(len(logits))
        rank_value = float(torch.where(idx_list == idx)[0])
        rank_values.append(float(rank_value))
    medrank_estimate, _, _, _ = jackknife.jackknife_stats(np.asarray(rank_values), np.median, 0.95)
    return medrank_estimate


def tag_to_binary(tag_list, list_of_label, tag_to_idx):
    bainry = np.zeros([len(list_of_label),], dtype=np.float32)
    for tag in tag_list:
        bainry[tag_to_idx[tag]] = 1.0
    return bainry

def _continuous_target(binary):
    sum_= binary.sum(dim=1)
    mask = sum_ - torch.matmul(binary, binary.T)
    inverse_mask = 1 / mask
    continuous_label = torch.nan_to_num(inverse_mask, posinf=.0)
    return continuous_label