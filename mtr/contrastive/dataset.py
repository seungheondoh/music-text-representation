import os
import json
import random
import pickle
import numpy as np
import pandas as pd
import torch
from typing import Callable, List, Dict, Any
from torch.utils.data import Dataset

class ECALS_Dataset(Dataset):
    def __init__(self, data_path, split, sr, duration, num_chunks, text_preprocessor=None, text_type="bert", text_rep="stochastic"):
        self.data_path = data_path
        self.split = split
        self.sr = sr
        self.text_preprocessor = text_preprocessor
        self.input_length = int(sr * duration)
        self.num_chunks = num_chunks
        self.text_type = text_type
        self.text_rep = text_rep
        self.msd_to_id = pickle.load(open(os.path.join(data_path, "lastfm_annotation", "MSD_id_to_7D_id.pkl"), 'rb'))
        self.id_to_path = pickle.load(open(os.path.join(data_path, "lastfm_annotation", "7D_id_to_path.pkl"), 'rb'))
        self.get_split()
        self.get_file_list()
    
    def get_split(self):
        track_split = json.load(open(os.path.join(self.data_path, "ecals_annotation", "ecals_track_split.json"), "r"))
        self.train_track = track_split['train_track'] + track_split['extra_track']
        self.valid_track = track_split['valid_track']
        self.test_track = track_split['test_track']
    
    def get_file_list(self):
        annotation = json.load(open(os.path.join(self.data_path, "ecals_annotation", "annotation.json"), 'r'))
        self.list_of_label = json.load(open(os.path.join(self.data_path, "ecals_annotation", "ecals_tags.json"), 'r'))
        self.tag_to_idx = {i:idx for idx, i in enumerate(self.list_of_label)}
        if self.split == "TRAIN":
            self.fl = [annotation[i] for i in self.train_track]
        elif self.split == "VALID":
            self.fl = [annotation[i] for i in self.valid_track]
        elif self.split == "TEST":
            self.fl = [annotation[i] for i in self.test_track]
        else:
            raise ValueError(f"Unexpected split name: {self.split}")
        del annotation
    
    def audio_load(self, msd_id):
        audio_path = self.id_to_path[self.msd_to_id[msd_id]]
        audio = np.load(os.path.join(self.data_path, "npy", audio_path.replace(".mp3",".npy")), mmap_mode='r')
        random_idx = random.randint(0, audio.shape[-1]-self.input_length)
        audio = torch.from_numpy(np.array(audio[random_idx:random_idx+self.input_length]))
        return audio

    def tag_to_binary(self, tag_list):
        bainry = np.zeros([len(self.list_of_label),], dtype=np.float32)
        for tag in tag_list:
            bainry[self.tag_to_idx[tag]] = 1.0
        return bainry

    def text_load(self, tag_list):
        """
        input:  tag_list = list of tag
        output: text = string of text
        """
        if self.text_rep == "caption":
            if self.split == "TRAIN":
                random.shuffle(tag_list)
            text = tag_list
        elif self.text_rep == "tag":
            text = [random.choice(tag_list)]
        elif self.text_rep == "stochastic":
            k = random.choice(range(1, len(tag_list)+1)) 
            text = random.sample(tag_list, k)
        return text

    def get_train_item(self, index):
        item = self.fl[index]
        tag_list = item['tag']
        binary = self.tag_to_binary(tag_list)
        text = self.text_load(tag_list)
        audio_tensor = self.audio_load(item['track_id'])
        return {
            "audio":audio_tensor, 
            "binary":binary, 
            "text":text
            }

    def get_eval_item(self, index):
        item = self.fl[index]
        tag_list = item['tag']
        binary = self.tag_to_binary(tag_list)
        text = self.text_load(tag_list)
        tags = self.list_of_label
        track_id = item['track_id']
        audio_path = self.id_to_path[self.msd_to_id[track_id]]
        audio = np.load(os.path.join(self.data_path, "npy", audio_path.replace(".mp3",".npy")), mmap_mode='r')
        hop = (len(audio) - self.input_length) // self.num_chunks
        audio = np.stack([np.array(audio[i * hop : i * hop + self.input_length]) for i in range(self.num_chunks)]).astype('float32')
        return {
            "audio":audio, 
            "track_id":track_id, 
            "tags":tags, 
            "binary":binary, 
            "text":text
        }

    def __getitem__(self, index):
        if (self.split=='TRAIN') or (self.split=='VALID'):
            return self.get_train_item(index)
        else:
            return self.get_eval_item(index)

    def batch_processor(self, batch):
        # batch = list of dcitioanry
        audio = [item_dict['audio'] for item_dict in batch]
        binary = [item_dict['binary'] for item_dict in batch]
        audios = torch.stack(audio)
        binarys = torch.tensor(np.stack(binary))
        text, text_mask = self._text_preprocessor(batch, "text")
        return {"audio":audios, "binary":binarys, "text":text, "text_mask":text_mask}

    def _text_preprocessor(self, batch, target_text):
        if self.text_type == "bert":
            batch_text = [", ".join(item_dict[target_text]) for item_dict in batch]
            encoding = self.text_preprocessor.batch_encode_plus(batch_text, padding='longest', max_length=64, truncation=True, return_tensors="pt")
            text = encoding['input_ids']
            text_mask = encoding['attention_mask']
        elif self.text_type == "glove":
            batch_emb = []
            batch_text = [item_dict[target_text] for item_dict in batch]
            for tag_seq in batch_text:
                tag_seq_emb = [np.array(self.text_preprocessor[token]).astype('float32') for token in tag_seq]
                batch_emb.append(torch.from_numpy(np.mean(tag_seq_emb, axis=0)))
            text = torch.stack(batch_emb)
            text_mask = None    
        return text, text_mask
            
    def __len__(self):
        return len(self.fl)
