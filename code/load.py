import os

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

import json



def load_data(args):

    messages = read_data(args)

    # Split the data into training and testing sets
    train_df, test_df = train_test_split(messages, test_size=0.2, random_state=args.seed)
    # Split the training data again to create a validation set
    train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=args.seed)

    # Convert the dataframes into Dataset objects
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    val_dataset = Dataset.from_pandas(val_df)


    # Combine the datasets into a DatasetDict
    dataset = DatasetDict({
        'train': train_dataset,
        'test': test_dataset,
        'validation': val_dataset
    })

    return dataset

def read_data(args):    
    data_path = os.path.join(args.input_dir, f'{args.dataset}.json')
    
    data = json.load(open(data_path))
    messages = pd.DataFrame(data)
    
    return messages


def load_tokenizer(args):
    # task1: load bert tokenizer from pretrained "bert-base-uncased", you can also set truncation_side as "left"
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', truncation=True, truncation_side='left')
    return tokenizer

if __name__ == '__main__':
    from arguments import params
    
    args = params()
    
    dataset = load_data(args)  

    # Count unique authors in the training dataset
    train_data = dataset['train']
    # Assuming the author column is named 'author'. Replace with actual column name if different
    if 'author' in train_data.column_names:
        unique_authors = len(set(train_data['author']))
        print(f"Number of unique authors in the training dataset: {unique_authors}")
    else:
        print("Column 'author' not found in the dataset. Available columns:", train_data.column_names)