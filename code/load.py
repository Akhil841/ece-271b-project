import os

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer


def load_data(args):

    messages, num_authors = set_up_csv(args)
    messages = clean_data(args, messages)

    # Split the data into training and testing sets
    train_df, test_df = train_test_split(messages, test_size=0.2, random_state=args.seed)
    # Split the training data again to create a validation set
    train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=args.seed)

    # Convert the dataframes into Dataset objects
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    val_dataset = Dataset.from_pandas(val_df)

    #No use for now
    #less_than_dataset = Dataset.from_pandas(less_than)

    # Combine the datasets into a DatasetDict
    dataset = DatasetDict({
        'train': train_dataset,
        'test': test_dataset,
        'validation': val_dataset
    })

    return dataset, num_authors

def set_up_csv(args):

    data_path = os.path.join(args.input_dir, f'{args.dataset}.csv')

    messages = pd.read_csv(data_path)

    #Remove messages that are under 3 words
    messages = messages[messages['text'].apply(lambda x: len(str(x).split()) >= 3)]

    #remove authors with less than min messages
    author_counts = messages['author'].value_counts()
    less_than_authors = author_counts[author_counts < args.min_messages].index
    messages = messages[~messages['author'].isin(less_than_authors)]

    # Group by author and take a random sample of min_messages from each group
    messages = messages.groupby('author').apply(lambda x: x.sample(n=min(len(x), args.min_messages))).reset_index(drop=True)

    num_authors = messages['author'].nunique()
    print(f"Number of authors: {num_authors}")

    #number of messages per author
    author_counts = messages['author'].value_counts()
    print(f"Number of messages per author: {author_counts}")



    #print(f"Number of authors: {num_authors}")
    
    #number of messages per author
    author_counts = messages['author'].value_counts()
    #print(f"Number of messages per author: {author_counts}")
    
    
    
    # Add an ID to each message
    messages['id'] = range(len(messages))
    # Encode the author names into numerical labels
    label_encoder = LabelEncoder()
    messages['label'] = label_encoder.fit_transform(messages['author'])
    # Rename the 'author' column to 'label_text'
    messages = messages.rename(columns={'author': 'label_text'})

    return messages, num_authors

def clean_data(args, messages):
    # Check for rows with null values in the 'text' column
    null_text_rows = messages['text'].isnull()
    if null_text_rows.any():
        messages = messages.dropna(subset=['text'])


    return messages


def load_tokenizer(args):
    # task1: load bert tokenizer from pretrained "bert-base-uncased", you can also set truncation_side as "left"
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', truncation=True, truncation_side='left')
    return tokenizer