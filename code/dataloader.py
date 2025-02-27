import os
import pdb
import pickle as pkl
import pprint
import random
import sys

import numpy as np
import torch
from datasets import DatasetDict
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm import tqdm
from tqdm import tqdm as progress_bar
from transformers import BertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lm_model = BertModel.from_pretrained('bert-base-uncased').to(device)

def get_dataloader(args, dataset, split='train'):
    sampler = RandomSampler(dataset) if split == 'train' else SequentialSampler(dataset)
    collate = dataset.collate_func

    b_size = args.batch_size
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=b_size, collate_fn=collate)
    print(f"Loaded {split} data with {len(dataloader)} batches")
    return dataloader

def get_word_embeddings(input_ids):
    embeddings = lm_model.get_input_embeddings()(input_ids)
    return embeddings


def prepare_inputs(batch):
    """
        This function converts the batch of variables to input_ids, token_type_ids and attention_mask which the
        BERT encoder requires. It also separates the targets (ground truth labels) for supervised-loss.
    """
    # 0: input_ids, 1: token_type_ids, 2: attention_mask, 3: target/label 4: text label

    left_input = {
        'input_ids': batch[0].to(device),
        'token_type_ids': batch[1].to(device),
        'attention_mask': batch[2].to(device)
    }
    right_input = {
        'input_ids': batch[3].to(device),
        'token_type_ids': batch[4].to(device),
        'attention_mask': batch[5].to(device)
    }
    labels = batch[6].to(device)
    return (left_input, right_input), labels


import torch
import random

    
def check_cache(args):
    folder = 'cache'
    cache_path = os.path.join(args.input_dir, folder, f'{args.dataset}.csv')
    use_cache = not args.ignore_cache

    if os.path.exists(cache_path) and use_cache:
        print(f'Loading features from cache at {cache_path}')
        results = pkl.load( open( cache_path, 'rb' ) )
        return results, True
    else:
        print(f'Creating new input features ...')
        return cache_path, False

def prepare_features(args, data, tokenizer, cache_path):
    all_features = {}

    for split, examples in data.items():
        feats = []
        # task1: process examples using tokenizer. Wrap it using BaseInstance class and append it to feats list.
        for example in progress_bar(examples, total=len(examples)):
            # tokenizer: set padding to 'max_length', set truncation to True, set max_length to args.max_len
            embed_data1 = tokenizer(example['text1'], padding='max_length', truncation=True, max_length=args.max_len)
            embed_data2 = tokenizer(example['text2'], padding='max_length', truncation=True, max_length=args.max_len)
            instance = BaseInstance(embed_data1, embed_data2, example)
            feats.append(instance)
            
        all_features[split] = feats


    pkl.dump(all_features, open(cache_path, 'wb'))
    return all_features

def process_data(args, features, tokenizer):

  datasets = {}
  for split, feat in features.items():
      ins_data = feat
      datasets[split] = IntentDataset(ins_data, tokenizer, split)

  return datasets



class BaseInstance(object):
    def __init__(self, embed_data1, embed_data2, example):
        # inputs to the transformer
        self.embedding1 = embed_data1['input_ids']
        self.segments1 = embed_data1['token_type_ids']
        self.input_mask1 = embed_data1['attention_mask']
        
        self.embedding2 = embed_data2['input_ids']
        self.segments2 = embed_data2['token_type_ids']
        self.input_mask2 = embed_data2['attention_mask']

        # labels
        self.intent_label = example['label']

        # for references
        self.text1 = example['text1']   # in natural language text
        self.text2 = example['text2']   # in natural language text

class IntentDataset(Dataset):
    def __init__(self, data, tokenizer, split='train'):
        self.data = data
        self.tokenizer = tokenizer
        self.split = split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_func(self, batch):
        input_ids1 = torch.tensor([f.embedding1 for f in batch], dtype=torch.long)
        segment_ids1 = torch.tensor([f.segments1 for f in batch], dtype=torch.long)
        input_masks1 = torch.tensor([f.input_mask1 for f in batch], dtype=torch.long)
        
        
        input_ids2 = torch.tensor([f.embedding2 for f in batch], dtype=torch.long)
        segment_ids2 = torch.tensor([f.segments2 for f in batch], dtype=torch.long)
        input_masks2 = torch.tensor([f.input_mask2 for f in batch], dtype=torch.long)
        
        label_ids = torch.tensor([f.intent_label for f in batch], dtype=torch.long)

        return input_ids1, segment_ids1, input_masks1, input_ids2, segment_ids2, input_masks2, label_ids

