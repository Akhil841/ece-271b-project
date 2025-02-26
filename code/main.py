import math
import os
import pickle
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from gensim import corpora, models
from joblib import dump, load
from torch import nn
from torch.optim.swa_utils import SWALR, AveragedModel, update_bn
from tqdm import tqdm as progress_bar

from arguments import params
from dataloader import (
    DataLoader,
    TopicDistributionDataset,
    check_cache,
    get_dataloader,
    prepare_data_lda,
    prepare_features,
    prepare_inputs,
    process_data,
)
from load import load_data, load_tokenizer
from loss import SupConLoss
from model import LDA, BaselineModel, Contrastive, Ensemble, GRUNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def baseline_train(args, model, datasets, tokenizer, num_authors):

    CHECKPOINT = 'ckpt_mdl_{}_ep_{}_hsize_{}_dout_{}'.format(args.task, args.n_epochs, args.hidden_dim, args.drop_rate)
    train_dataloader = get_dataloader(args, datasets['train'], 'train')

    model.optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    model.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model.optimizer, args.n_epochs)

    train_accuracies = []
    validation_accuracies = []

    criterion = criterion = nn.CrossEntropyLoss()

    for epoch_count in range(args.n_epochs):
        losses = 0
        model.train()

        acc = 0
        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
            inputs, labels = prepare_inputs(batch, args.task)
            logits = model(inputs, labels)
            loss = criterion(logits, labels)
            loss.backward()

            tem = (logits.argmax(1) == labels).float().sum()
            acc += tem.item()

            model.optimizer.step()  # backprop to update the weights
            model.scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            losses += loss.item()

        train_accuracies.append(acc/len(datasets['train']))
        validation_accuracies.append(run_eval(args, model, datasets, tokenizer, num_authors, split='validation'))
        print('training epoch', epoch_count, '| losses:', losses, '| accuracy:', acc/len(datasets['train']))

        if not os.path.isdir('models'):
            os.mkdir('models')

        # Save checkpoint.
        if (epoch_count % args.save_every == 0 and epoch_count != 0)  or epoch_count == args.n_epochs - 1:
            print('=======>Saving..')
            torch.save({
                'epoch': epoch_count + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model.optimizer.state_dict(),
                'loss': loss,
                }, './models/' + CHECKPOINT + '.t%s' % epoch_count)

    graph(args, train_accuracies, validation_accuracies)



def run_eval(args, model, datasets, tokenizer, num_authors, split='validation'):
    model.eval()
    dataloader = get_dataloader(args, datasets[split], split)

    acc = 0
    loss = 0
    criterion = nn.CrossEntropyLoss()
    confusion_matrix = np.zeros((num_authors, num_authors))

    for step, batch in progress_bar(enumerate(dataloader), total=len(dataloader)):
        inputs, labels = prepare_inputs(batch, args.task)
        logits = model(inputs, labels)
        loss += criterion(logits, labels).item()

        # Update the confusion matrix
        predictions = logits.argmax(1)

        for i in range(len(labels)):

            #print('labels[i].item()', labels[i].item())
            confusion_matrix[labels[i].item(), predictions[i].item()] += 1

        tem = (logits.argmax(1) == labels).float().sum()
        acc += tem.item()


    # Calculate the accuracy for each category and the most likely misclassification
    category_acc = {i: confusion_matrix[i, i] / np.sum(confusion_matrix[i, :]) for i in range(num_authors)}
    most_likely_misclassifications = {}
    for i in range(num_authors):
        confusion_matrix[i, i] = -1  # Temporarily set the diagonal element to -1
        most_likely_misclassifications[i] = np.argmax(confusion_matrix[i, :])
        confusion_matrix[i, i] = category_acc[i] * np.sum(confusion_matrix[i, :])  # Restore the diagonal element

    print(f'\n{split} acc:', acc/len(datasets[split]), f'loss {loss}', f'|dataset split {split} size:', len(datasets[split]))
    print(f'Category accuracies: {category_acc}')
    print(f'Most likely misclassifications: {most_likely_misclassifications}\n')
    return acc/len(datasets[split])




if __name__ == "__main__":
  args = params()
  args = setup_gpus(args)
  args = check_directories(args)
  set_seed(args)

  cache_results, already_exist = check_cache(args)
  tokenizer = load_tokenizer(args)

  if already_exist:
    features = cache_results
    data, num_authors = load_data(args)
  else:
    data, num_authors = load_data(args)
    features = prepare_features(args, data, tokenizer, cache_results)

  datasets = process_data(args, features, tokenizer)

  print('Data loaded and processed')

  print('Training model')
  if args.task == 'dl':
    model = BaselineModel(args, tokenizer, target_size=num_authors).to(device)
    
    run_eval(args, model, datasets, tokenizer, num_authors, split='validation')
    run_eval(args, model, datasets, tokenizer, num_authors, split='test')
    baseline_train(args, model, datasets, tokenizer, num_authors)
    run_eval(args, model, datasets, tokenizer, num_authors,num_authors, split='test')
  else:
    raise ValueError(f'Invalid task: {args.task}')