## Toward Universal Text-to-Music Retrieval
- [Demo](https://seungheondoh.github.io/text-music-representation-demo/)
- [Paper ArXiv](#)
- [Pretrained Model at Zenodo](#)

<p align = "center">
<img src = "https://i.imgur.com/Og18FbB.png">
</p>

### Introduction
This is a PyTorch implementation of [Toward Universal Text-to-Music Retrieval](#) for multi-modal music representation.

### Main Results
The following results are based on [MSD-ECAL](https://github.com/SeungHeonDoh/msd-subsets) pre-training.
**Pre-trained models** and **configs** can be found at [Zenodo-Pretrained]().

<table>
<thead>
  <tr>
    <th></th>
    <th></th>
    <th></th>
    <th colspan="2">Tag based Retrieval</th>
    <th colspan="5">Language based Retrieval</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="2">Model Type</td>
    <td rowspan="2">Text Enc.</td>
    <td rowspan="2">Text Rep.</td>
    <td>50 Tag</td>
    <td>1054 Tag</td>
    <td colspan="5">1000 Caption</td>
  </tr>
  <tr>
    <td>ROC/PR</td>
    <td>ROC/PR</td>
    <td>R@1</td>
    <td>R@5</td>
    <td>R@10</td>
    <td>mAP</td>
    <td>MedR</td>
  </tr>
  <tr>
    <td>Classification</td>
    <td>Binary</td>
    <td>Tag</td>
    <td>90.5 / 39.8</td>
    <td>86.2 / 8.9</td>
    <td>3.7</td>
    <td>14.3</td>
    <td>22.5</td>
    <td>8.4</td>
    <td>64</td>
  </tr>
  <tr>
    <td>Triplet</td>
    <td>GloVe</td>
    <td>Tag</td>
    <td>89.2 / 36.0</td>
    <td>82.6 / 6.1</td>
    <td>2.8</td>
    <td>11.2</td>
    <td>18.6</td>
    <td>6.6</td>
    <td>51.5</td>
  </tr>
  <tr>
    <td>Triplet</td>
    <td>GloVe</td>
    <td>Caption</td>
    <td>88.6 / 37.1</td>
    <td>76.8 / 5.3</td>
    <td>5.4</td>
    <td>22.1</td>
    <td>35.0</td>
    <td>13.0</td>
    <td>17.0</td>
  </tr>
  <tr>
    <td>Triplet</td>
    <td>GloVe</td>
    <td>Stochastic</td>
    <td>89.2 / 37.6</td>
    <td>81.6 / 6.2</td>
    <td>6.4</td>
    <td>21.8</td>
    <td>32.7</td>
    <td>12.8</td>
    <td>19.5</td>
  </tr>
  <tr>
    <td>Triplet</td>
    <td>BERT</td>
    <td>Tag</td>
    <td>86.9 / 30.2</td>
    <td>81.7 / 5.1</td>
    <td>1.6</td>
    <td>6.2</td>
    <td>12.0</td>
    <td>3.9</td>
    <td>68.0</td>
  </tr>
  <tr>
    <td>Triplet</td>
    <td>BERT</td>
    <td>Caption</td>
    <td>87.7 / 35.0</td>
    <td>78.9 / 5.4</td>
    <td>6.7</td>
    <td>23.6</td>
    <td>36.6</td>
    <td>14.1</td>
    <td>16.0</td>
  </tr>
  <tr>
    <td>Triplet</td>
    <td>BERT</td>
    <td>Stochastic</td>
    <td>88.4 / 35.0</td>
    <td>83.6 / 6.3</td>
    <td>6.6</td>
    <td>25.1</td>
    <td>39.4</td>
    <td>14.6</td>
    <td>16.0</td>
  </tr>
  <tr>
    <td>Contrastive</td>
    <td>BERT</td>
    <td>Tag</td>
    <td>90.6 / 40.2</td>
    <td>86.4 / 8.8</td>
    <td>2.5</td>
    <td>13.7</td>
    <td>22.5</td>
    <td>7.4</td>
    <td>47.0</td>
  </tr>
  <tr>
    <td>Contrastive</td>
    <td>BERT</td>
    <td>Caption</td>
    <td>87.0 / 32.5</td>
    <td>77.6 / 5.1</td>
    <td>6.8</td>
    <td>25.4</td>
    <td>38.4</td>
    <td>15.3</td>
    <td>17.0</td>
  </tr>
  <tr>
    <td>Contrastive</td>
    <td>BERT</td>
    <td>Stochastic</td>
    <td>89.8 / 38.0</td>
    <td>84.8 / 7.7</td>
    <td>10.2</td>
    <td>29.8</td>
    <td>42.8</td>
    <td>18.7</td>
    <td>13.0</td>
  </tr>
</tbody>
</table>


### Text Representation
From our empirical study, we find that there is a strong association between text representation (train stage) and text query types (test stage). We propose a stochastic text representation. During the training stage, we select K words from L length text caption. At this time, K is uniformly randomly sampled among integer numbers from 1 (tag length) to L (caption length). Unlike the dropout method, which determines the length by probability value, stochastic sampling has a dynamic input length.

```
def text_load(self, tag_list):
    """
    input:  tag_list = list of tag
    output: text = string of text
    """
    if self.text_rep == "caption":
        if self.split == "TRAIN":
            random.shuffle(tag_list)
        text = ", ".join(tag_list)
    elif self.text_rep == "tag":
        text = [random.choice(tag_list)]
    elif self.text_rep == "stochastic":
        k = random.choice(range(1, len(tag_list)+1)) 
        sampled_tag_list = random.sample(tag_list, k)
        text = ", ".join(sampled_tag_list)
    return text
```

### Requirements

1. Install python and PyTorch:
    - python==3.8
    - torch==1.12.1 (Please install it according to your [CUDA version](https://pytorch.org/get-started/previous-versions/).)
    
2. Other requirements:
    - pip install -r requirements.txt

```
conda create -n YOUR_ENV_NAME python=3.8
conda activate YOUR_ENV_NAME
pip install -r requirements.txt
```

### Using Pretrained Model
```
wget 
tar -zxvf mtr.tar.gz 
```

### Start Training

1. Download ECALS(Extended Cleaned tag and Artist-Level Stratified split) dataset & MSD audio [Link](https://github.com/SeungHeonDoh/msd-subsets)

2. Pretraining (Quick start: mtr/contrastive/main.sh)
```
cd mtr/{triplet or contrastive}
python train.py --text_type {bert,glove} --text_rep {tag,caption,stochastic} --data_dir {msd-subsets} --multiprocessing-distributed
python test.py --text_type {bert,glove} --text_rep {tag,caption,stochastic} --data_dir {msd-subsets}
```

Following [MoCo V3 Repo](https://github.com/facebookresearch/moco-v3), This repo only *multi-gpu*, *DistributedDataParallel* training is supported; single-gpu or DataParallel training is not supported. This code is improved to better suit the *multi-node* setting. 

other pretrining settings are:
```
parser.add_argument("--duration", default=9.91, type=int)
parser.add_argument("--sr", default=16000, type=int)
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
parser.add_argument("--mlp_dim", default=128, type=int) -> joint embedding dim
```


3. Zeroshot Transfer, and Probing (Quick start: mtr/transfer/main.sh)
```
cd mtr/transfer
python extractor.py --framework {classification, triplet, contrastive} --text_type {binary, glove, bert} --text_rep {tag,caption,stocahstic} --eval_dataset $DATASET

python eval_zs.py --framework {triplet, contrastive} --text_type {binary, glove, bert} --text_rep {tag,caption,stocahstic} --eval_dataset $DATASET

python train_probing.py --probe_type {linear, mlp} --framework {classification, triplet, contrastive} --text_type {binary, glove, bert} --text_rep {tag,caption,stocahstic} --eval_dataset $DATASET

python eval_probing.py --probe_type {linear, mlp} --framework {classification, triplet, contrastive} --text_type {binary, glove, bert} --text_rep {tag,caption,stocahstic} --eval_dataset $DATASET
```

