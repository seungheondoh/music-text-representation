import os
import json
import random
import pickle
import numpy as np
import pandas as pd
import torch
from typing import Callable, List, Dict, Any
from torch.utils.data import Dataset

class FMA_Dataset(Dataset):
    def __init__(self, data_path, split, sr, duration, num_chunks):
        self.data_path = data_path
        self.split = split
        self.sr = sr
        self.input_length = int(sr * duration)
        self.num_chunks = num_chunks
        self.black_list = [99134, 108925, 133297]
        self.get_split()
        self.get_file_list()
    
    def get_split(self):
        track_split = json.load(open(os.path.join(self.data_path, "fma", "track_split.json"), "r"))
        self.train_track = track_split['train_track']
        self.valid_track = track_split['valid_track']
        self.test_track = track_split['test_track']
    
    def get_file_list(self):
        annotation = json.load(open(os.path.join(self.data_path, "fma", "annotation.json"), 'r'))
        self.list_of_label = json.load(open(os.path.join(self.data_path, "fma", "fma_tags.json"), 'r'))
        self.tag_to_idx = {i:idx for idx, i in enumerate(self.list_of_label)}
        if self.split == "TRAIN":
            self.fl = [annotation[str(i)] for i in self.train_track if int(i) not in self.black_list]
        elif self.split == "VALID":
            self.fl = [annotation[str(i)] for i in self.valid_track if int(i) not in self.black_list]
        elif self.split == "TEST":
            self.fl = [annotation[str(i)] for i in self.test_track if int(i) not in self.black_list]
        elif self.split == "ALL":
            self.fl = [v for k,v in annotation.items() if int(k) not in self.black_list]
        else:
            raise ValueError(f"Unexpected split name: {self.split}")
        del annotation

    def get_numpy_path(self, track_id):
        tid_str = '{:06d}'.format(int(track_id))
        return os.path.join(tid_str[:3], tid_str + '.npy')

    def audio_load(self, track_id):
        npy_path = self.get_numpy_path(track_id)
        audio = np.load(os.path.join(self.data_path, "fma", "npy", npy_path), mmap_mode='r')
        random_idx = random.randint(0, audio.shape[-1]-self.input_length)
        audio = torch.from_numpy(np.array(audio[random_idx:random_idx+self.input_length]))
        return audio

    def tag_to_binary(self, text):
        bainry = np.zeros([len(self.list_of_label),], dtype=np.float32)
        if isinstance(text, str):
            bainry[self.tag_to_idx[text]] = 1.0
        elif isinstance(text, list):
            for tag in text:
                bainry[self.tag_to_idx[tag]] = 1.0
        return bainry

    def get_train_item(self, index):
        item = self.fl[index]
        tag_list = item['tag']
        binary = self.tag_to_binary(tag_list)
        audio_tensor = self.audio_load(str(item['track_id']))
        return {
            "audio":audio_tensor, 
            "binary":binary
            }

    def get_eval_item(self, index):
        item = self.fl[index]
        tag_list = item['tag']
        binary = self.tag_to_binary(tag_list)
        text = ", ".join(tag_list)
        tags = self.list_of_label
        track_id = str(item['track_id'])
        npy_path = self.get_numpy_path(track_id)
        audio = np.load(os.path.join(self.data_path, "fma", "npy", npy_path), mmap_mode='r')
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
            
    def __len__(self):
        return len(self.fl)
