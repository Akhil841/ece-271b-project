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

    batch = make_pairs(batch)

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

import torch
import random

def make_pairs(batch):
    '''
    Takes in batch of data indexed as 0: input_ids, 1: token_type_ids, 2: attention_mask, 3: target/label, 4: text label
    
    Creates pairs of data for contrastive loss by creating pairs of data, with new labels of 1 if the data is from the same class, and -1 if the data is from different classes.
    Balances the number of positive and negative pairs such that the total number of pairs equals the original batch size.
    
    Return batch of data indexed as:
    0: input_ids of first, 1: token_type_ids of first, 2: attention_mask of first,
    3: input_ids of second, 4: token_type_ids of second, 5: attention_mask of second,
    6: label (1 if same class, -1 if different)
    '''
    pos_pairs = []
    neg_pairs = []
    batch_size = len(batch[0])
    
    # Create all possible pairs (i, j) with i < j
    for i in range(batch_size):
        for j in range(i+1, batch_size):
            # Compare target labels (assumes each label is a tensor scalar or similar)
            if batch[3][i].item() == batch[3][j].item():
                pos_pairs.append((i, j))
            else:
                neg_pairs.append((i, j))
    
    # Determine required number of pairs from each category
    required_pos = batch_size // 2
    required_neg = batch_size - required_pos

    # Sample positive pairs; use replacement if not enough
    if len(pos_pairs) >= required_pos:
        selected_pos = random.sample(pos_pairs, required_pos)
    else:
        selected_pos = random.choices(pos_pairs, k=required_pos)
    
    # Sample negative pairs; use replacement if not enough
    if len(neg_pairs) >= required_neg:
        selected_neg = random.sample(neg_pairs, required_neg)
    else:
        selected_neg = random.choices(neg_pairs, k=required_neg)
    
    all_pairs = selected_pos + selected_neg
    
    left_input_ids = []
    left_token_type_ids = []
    left_attention_mask = []
    right_input_ids = []
    right_token_type_ids = []
    right_attention_mask = []
    pair_labels = []
    
    for i, j in all_pairs:
        left_input_ids.append(batch[0][i])
        left_token_type_ids.append(batch[1][i])
        left_attention_mask.append(batch[2][i])
        
        right_input_ids.append(batch[0][j])
        right_token_type_ids.append(batch[1][j])
        right_attention_mask.append(batch[2][j])
        
        # Label: 1 if same class, 0 if different
        label = 1 if batch[3][i].item() == batch[3][j].item() else 0
        pair_labels.append(label)
    
    return [
        torch.stack(left_input_ids),
        torch.stack(left_token_type_ids),
        torch.stack(left_attention_mask),
        torch.stack(right_input_ids),
        torch.stack(right_token_type_ids),
        torch.stack(right_attention_mask),
        torch.tensor(pair_labels)
    ]
    
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
            embed_data = tokenizer(example['text'], padding='max_length', truncation=True, max_length=args.max_len)
            instance = BaseInstance(embed_data, example)
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
    def __init__(self, embed_data, example):
        # inputs to the transformer
        self.embedding = embed_data['input_ids']
        self.segments = embed_data['token_type_ids']
        self.input_mask = embed_data['attention_mask']

        # labels
        self.intent_label = example['label']

        # for references
        self.text = example['text']   # in natural language text
        self.label_text = example['label_text']

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
        input_ids = torch.tensor([f.embedding for f in batch], dtype=torch.long)
        segment_ids = torch.tensor([f.segments for f in batch], dtype=torch.long)
        input_masks = torch.tensor([f.input_mask for f in batch], dtype=torch.long)
        label_ids = torch.tensor([f.intent_label for f in batch], dtype=torch.long)

        label_texts = [f.label_text for f in batch]
        return input_ids, segment_ids, input_masks, label_ids, label_texts
        # return input_ids, label_ids, label_texts


def convert_data(dataset):
    return [{'id': item['id'], 'text': item['text'], 'label': item['label'], 'label_text': item['label_text']} for item in dataset]