import os
import pdb
import pickle as pkl
import pprint
import random
import sys

import numpy as np
import torch
from datasets import DatasetDict
from gensim import corpora, models
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm import tqdm
from tqdm import tqdm as progress_bar
from transformers import BertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloader(args, dataset, split='train'):
    sampler = RandomSampler(dataset) if split == 'train' else SequentialSampler(dataset)
    collate = dataset.collate_func

    b_size = args.batch_size
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=b_size, collate_fn=collate)
    print(f"Loaded {split} data with {len(dataloader)} batches")
    return dataloader

lm_model = BertModel.from_pretrained('bert-base-uncased').to(device)

def get_word_embeddings(input_ids):
    embeddings = lm_model.get_input_embeddings()(input_ids)
    return embeddings


def prepare_inputs(batch, model, use_text=False):
    """
        This function converts the batch of variables to input_ids, token_type_ids and attention_mask which the
        BERT encoder requires. It also separates the targets (ground truth labels) for supervised-loss.
    """

    if model == 'lda':

        btt = [b.to(device) for b in batch[3:5]]
        inputs = {'topic_distribution': btt[1]}
        targets = btt[0]

        return inputs, targets
    elif model == 'baseline' or model == 'contrastive':
        btt = [b.to(device) for b in batch[:4]]
        inputs = {'input_ids': btt[0], 'token_type_ids': btt[1], 'attention_mask': btt[2]}
        targets = btt[3]

        if use_text:
            target_text = batch[4]
            return inputs, targets, target_text
        else:
            return inputs, targets
    elif model == 'GRU':
        btt = [b.to(device) for b in [batch[0], batch[3]]]
        input_ids = btt[0]
        inputs = get_word_embeddings(input_ids)
        targets = btt[1]
        return inputs, targets

    elif model == "ensemble":
        inputs_gru, targets = prepare_inputs(batch, 'GRU')
        # inputs_baseline, targets = prepare_inputs(batch, 'baseline')
        inputs_lda, _ = prepare_inputs(batch, 'lda')
        inputs_contrastive, _ = prepare_inputs(batch, 'contrastive')
        return (inputs_gru, inputs_lda, inputs_contrastive), targets


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
  train_size, dev_size = len(features['train']), len(features['validation'])

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

class TopicDistributionDataset(Dataset):
    def __init__(self, data, encoder, dictionary, batch_size, split, tokenizer, ins_data):
        self.data = data
        self.encoder = encoder
        self.dictionary = dictionary
        self.batch_size = batch_size
        self.topic_distributions = self.get_all_topic_distributions(split)
        self.tokenizer = tokenizer
        self.ins_data = ins_data

    def get_all_topic_distributions(self, split):
        all_distributions = []
        for i in tqdm(range(0, len(self.data), self.batch_size), desc=f"Getting topic distributions for {split} data"):
            batch = [item['text'] for item in self.data[i:i+self.batch_size]]
            distributions = get_topic_distributions(self.encoder, self.dictionary, batch)
            all_distributions.extend(distributions)
        return all_distributions

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        item['topic_distribution'] = self.topic_distributions[idx]
        return item, self.ins_data[idx]

    def collate_func(self, batch):
        input_ids_lm = torch.tensor([f[1].embedding for f in batch], dtype=torch.long)
        segment_ids = torch.tensor([f[1].segments for f in batch], dtype=torch.long)
        input_masks = torch.tensor([f[1].input_mask for f in batch], dtype=torch.long)
        input_ids = torch.tensor([f[0]['id'] for f in batch], dtype=torch.long)
        label_ids = torch.tensor([f[0]['label'] for f in batch], dtype=torch.long)
        topic_distributions = torch.stack([(f[0]['topic_distribution']).clone().detach().requires_grad_(True) for f in batch])
        label_texts = [f[0]['label_text'] for f in batch]
        return input_ids_lm, segment_ids, input_masks, label_ids, topic_distributions, label_texts
        # return input_ids, label_ids, label_texts, topic_distributions

def convert_data(dataset):
    return [{'id': item['id'], 'text': item['text'], 'label': item['label'], 'label_text': item['label_text']} for item in dataset]