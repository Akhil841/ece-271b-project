import json
from collections import defaultdict
import random
import itertools
import torch
from transformers import BertTokenizer

def load(file_path):
    data = load_data(file_path)
    author_messages = group_messages_by_author(data)
    return author_messages
    
def create_training_data(author_messages, n_authors, n_pairs):
    eligible_authors = get_eligible_authors(author_messages, n_authors)
    chosen_authors = choose_authors(eligible_authors, n_authors)
    training_examples = generate_training_examples(author_messages, chosen_authors, n_pairs)
    return training_examples

def prepare_training_data(training_examples, tokenizer, max_length=512):
    texts1 = [ex[0] for ex in training_examples]
    texts2 = [ex[1] for ex in training_examples]
    labels = [ex[2] for ex in training_examples]

    encoded_inputs1, encoded_inputs2 = tokenize_texts(tokenizer, texts1, texts2, max_length)
    labels = torch.tensor(labels, dtype=torch.float)

    training_data = {
        "input_ids1": encoded_inputs1["input_ids"],
        "attention_mask1": encoded_inputs1["attention_mask"],
        "input_ids2": encoded_inputs2["input_ids"],
        "attention_mask2": encoded_inputs2["attention_mask"],
        "labels": labels,
        "author1": [ex[3] for ex in training_examples], 
        "author2": [ex[4] for ex in training_examples]
    }
    return training_data



def load_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def group_messages_by_author(data):
    author_messages = defaultdict(list)
    for item in data:
        author_messages[item['author']].append(item['body'])
        
    return author_messages

def get_eligible_authors(author_messages, n_authors):
    eligible_authors = [auth for auth, msgs in author_messages.items() if len(msgs) >= 2]
    if len(eligible_authors) < n_authors:
        raise ValueError(f"Not enough authors with at least 2 messages (found {len(eligible_authors)} but need {n_authors}).")
    return eligible_authors

def choose_authors(eligible_authors, n_authors):
    return random.sample(eligible_authors, n_authors)

def create_positive_pairs(msgs, n_pairs):
    all_pairs = list(itertools.combinations(msgs, 2))
    if n_pairs > len(all_pairs):
        raise ValueError("Not enough unique pairs can be formed from messages.")
    return random.sample(all_pairs, n_pairs)

def generate_training_examples(author_messages, chosen_authors, n_pairs):
    training_examples = []
    pos_pairs_mapping = {}
    author_texts_mapping = {}

    for author in chosen_authors:
        msgs = author_messages[author]
        pos_pairs = create_positive_pairs(msgs, n_pairs)
        pos_pairs_mapping[author] = pos_pairs
        agg_texts = [msg for pair in pos_pairs for msg in pair]
        author_texts_mapping[author] = agg_texts
        for pair in pos_pairs:
            training_examples.append((pair[0], pair[1], 1, author, author))

    for author in chosen_authors:
        own_texts = author_texts_mapping[author]
        other_texts = [(text, other) for other in chosen_authors if other != author for text in author_texts_mapping[other]]
        random.shuffle(own_texts)
        random.shuffle(other_texts)
        neg_n = min(n_pairs, len(own_texts), len(other_texts))
        for i in range(neg_n):
            other_text, other_author = other_texts[i]
            training_examples.append((own_texts[i], other_text, 0, author, other_author))

    random.shuffle(training_examples)
    return training_examples

def print_training_examples_overview(training_examples, chosen_authors):
    pos_count = sum(1 for ex in training_examples if ex[2] == 1)
    neg_count = sum(1 for ex in training_examples if ex[2] == 0)
    print("Chosen authors:", chosen_authors)
    print("Number of positive examples:", pos_count)
    print("Number of negative examples:", neg_count)
    print("Sample training examples:")
    for ex in training_examples[:5]:
        print(ex)

def tokenize_texts(tokenizer, texts1, texts2, max_length=512):
    encoded_inputs1 = tokenizer(texts1, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt", clean_up_tokenization_spaces=True)
    encoded_inputs2 = tokenizer(texts2, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt", clean_up_tokenization_spaces=True)
    return encoded_inputs1, encoded_inputs2


