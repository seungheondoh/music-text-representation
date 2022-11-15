import os
import json
import random
import pickle
import numpy as np
import pandas as pd
import torch
from typing import Callable, List, Dict, Any
from torch.utils.data import Dataset

class OPENMIC_Dataset(Dataset):
    def __init__(self, data_path, split, sr, duration, num_chunks):
        self.data_path = data_path
        self.split = split
        self.sr = sr
        self.input_length = int(sr * duration)
        self.num_chunks = num_chunks
        self.get_split()
        self.get_file_list()
    
    def get_split(self):
        track_split = json.load(open(os.path.join(self.data_path, "openmic", "track_split.json"), "r"))
        self.train_track = track_split['train_track']
        self.valid_track = track_split['valid_track']
        self.test_track = track_split['test_track']
    
    def get_file_list(self):
        annotation = json.load(open(os.path.join(self.data_path, "openmic", "annotation.json"), 'r'))
        self.list_of_label = json.load(open(os.path.join(self.data_path, "openmic", "openmic_tags.json"), 'r'))
        self.tag_to_idx = {i:idx for idx, i in enumerate(self.list_of_label)}
        if self.split == "TRAIN":
            self.fl = [annotation[str(i)] for i in self.train_track]
        elif self.split == "VALID":
            self.fl = [annotation[str(i)] for i in self.valid_track]
        elif self.split == "TEST":
            self.fl = [annotation[str(i)] for i in self.test_track]
        elif self.split == "ALL":
            self.fl = list(annotation.values())
        else:
            raise ValueError(f"Unexpected split name: {self.split}")
        del annotation

    def id_to_audio_filename(self, sample_key: str, ext='ogg'):
        return os.path.join(sample_key[:3], '%s.%s' % (sample_key, ext))

    def audio_load(self, track_id):
        audio = np.load(os.path.join(self.data_path, "openmic", "npy", track_id + ".npy"), mmap_mode='r')
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
        track_id = item['track_id']
        print(item)
        audio = np.load(os.path.join(self.data_path, "openmic", "npy", track_id + ".npy"), mmap_mode='r')
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
