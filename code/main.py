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
from topic_extraction.topics import get_topic_distributions, train_lda
from utils import check_directories, graph, set_seed, setup_gpus

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

def lda_train(args, model, datasets, num_authors, tokenizer):
    split = 'train'
    criterion = nn.CrossEntropyLoss()
    train_dataloader = get_dataloader(args, datasets[split], split)
    train_accuracies, validation_accuracies = [], []

    model.optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    model.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model.optimizer, args.n_epochs)

    train_accuracies = []

    CHECKPOINT = 'ckpt_mdl_{}_ep_{}_hsize_{}_dout_{}'.format(args.task, args.n_epochs, args.hidden_dim, args.drop_rate)

    for epoch_count in range(args.n_epochs):
        losses = 0
        model.train()

        acc = 0
        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
            inputs, labels = prepare_inputs(batch, args.task)
            logits = model(inputs, labels)
            print("###", torch.max(labels), torch.min(labels))
            print(logits.shape, labels.shape)
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
            torch.save({
                'epoch': epoch_count + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model.optimizer.state_dict(),
                'loss': loss,
                }, './models/' + CHECKPOINT )

    graph(args, train_accuracies, validation_accuracies)


def contrastive_learning_train(args, model, datasets, tokenizer):

    CHECKPOINT = 'ckpt_mdl_{}_ep_{}_hsize_{}_dout_{}'.format(args.task, args.n_epochs, args.hidden_dim, args.drop_rate)
    train_dataloader = get_dataloader(args, datasets['train'], 'train')

    model.optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    model.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model.optimizer, args.n_epochs)

    train_accuracies = []
    validation_accuracies = []

    supcon_loss = SupConLoss(temperature=args.temperature).to(device)
    criterion = criterion = nn.CrossEntropyLoss()

    for epoch_count in range(args.supcon_epochs):  # supcon_epochs might be different from n_epochs
      model.train()
      supcon_losses = 0
      for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
          inputs, labels = prepare_inputs(batch, args.task)
          embeddings = model(inputs, labels, contrastive=True)  # Ensure model outputs embeddings in this mode
        #   print(embeddings.shape, labels.view(-1, 1).shape)
          loss = supcon_loss(embeddings.unsqueeze(1), labels.view(-1, 1))
          loss.backward()
          model.optimizer.step()
          model.scheduler.step()
          model.zero_grad()
          supcon_losses += loss.item()
      print(f"Epoch {epoch_count}, SupCon Loss: {supcon_losses / len(train_dataloader)}")

    #freeze encoder params
    for param in model.encoder.parameters():
      param.requires_grad = False

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

def ensemble_train(args, model, datasets, tokenizer):

    CHECKPOINT = 'ckpt_ens_{}_ep_{}_hsize_{}_dout_{}'.format(args.task, args.n_epochs, args.hidden_dim, args.drop_rate)
    train_dataloader = get_dataloader(args, datasets['train'], 'train')

    model.optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    model.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model.optimizer, args.n_epochs)

    train_accuracies = []
    validation_accuracies = []

    criterion = nn.CrossEntropyLoss()

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

def gru_train(args, model, datasets, tokenizer):
    CHECKPOINT = 'ckpt_gru_{}_ep_{}_hsize_{}_dout_{}'.format(args.task, args.n_epochs, args.hidden_dim, args.drop_rate)

    train_dataloader = get_dataloader(args, datasets['train'], 'train')

    model.optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    model.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model.optimizer, args.n_epochs)

    train_accuracies, validation_accuracies = [], []


    criterion = nn.CrossEntropyLoss()

    for epoch_count in range(args.n_epochs):
        model.train()
        losses, acc = 0, 0

        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
            inputs, labels = prepare_inputs(batch, args.task)
            logits = model(inputs, labels)
            loss = criterion(logits, labels)
            loss.backward()

            acc += (logits.argmax(1) == labels).float().sum().item()

            model.optimizer.step()
            model.scheduler.step()
            model.zero_grad()
            losses += loss.item()

        train_accuracy = acc / len(datasets['train'])
        validation_accuracy = run_eval(args, model, datasets, tokenizer, num_authors, split='validation')
        train_accuracies.append(train_accuracy)
        validation_accuracies.append(validation_accuracy)

        print(f'Training epoch {epoch_count} | Loss: {losses} | Train accuracy: {train_accuracy} | Validation accuracy: {validation_accuracy}')

        if (epoch_count % args.save_every == 0 and epoch_count != 0) or epoch_count == args.n_epochs - 1:
            if not os.path.isdir('models'):
                os.mkdir('models')
            torch.save({
                'epoch': epoch_count + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model.optimizer.state_dict(),
                'loss': loss,
            }, './models/' + CHECKPOINT + '.t%s' % epoch_count)


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
  if args.task == 'baseline':
    model = BaselineModel(args, tokenizer, target_size=num_authors).to(device)
    run_eval(args, model, datasets, tokenizer, num_authors, split='validation')
    run_eval(args, model, datasets, tokenizer, num_authors, split='test')
    baseline_train(args, model, datasets, tokenizer, num_authors)
    run_eval(args, model, datasets, tokenizer, num_authors,num_authors, split='test')
  elif args.task == 'contrastive':
    model = Contrastive(args, tokenizer, target_size=num_authors).to(device)
    # run_eval(args, model, datasets, tokenizer,num_authors, split='validation')
    # run_eval(args, model, datasets, tokenizer,num_authors, split='test')
    contrastive_learning_train(args, model, datasets, tokenizer)
    run_eval(args, model, datasets, tokenizer, num_authors, split='test')
  elif args.task == 'lda':
    lda_model_path = os.path.join('models', 'lda')
    dictionary_path = 'dictionary_59976'

    datasets = prepare_data_lda(data, features, args, tokenizer)
    print(datasets)
    model = LDA(args,target_size=num_authors).to(device)
    run_eval(args, model, datasets, tokenizer, num_authors, split='validation')
    run_eval(args, model, datasets, tokenizer, num_authors, split='test')
    lda_train(args, model, datasets, num_authors, tokenizer)
    run_eval(args, model, datasets, tokenizer, num_authors, split='test')

        # Save the datasets to the cache file
    print('Saving topic distribution datasets to cache')
    # with open(cache_file_path, 'wb') as f:
    #     pickle.dump(datasets, f)

    test = datasets['train'][0]
    print(test)


    model = LDA(args,target_size=4).to(device)
    # run_eval(args, model, datasets, tokenizer, split='validation')
    # run_eval(args, model, datasets, tokenizer, split='test')
    lda_train(args, model, datasets, num_authors, tokenizer)
    run_eval(args, model, datasets, tokenizer, split='test')

  elif args.task == 'ensemble':
    datasets_cache_path = os.path.join('assets', 'cache', 'datasets')
    if not os.path.exists(datasets_cache_path):
        os.makedirs(os.path.join('assets', 'cache'), exist_ok=True)
        datasets = prepare_data_lda(data, features, args, tokenizer)
        with open(datasets_cache_path, 'wb') as f:
            pickle.dump(datasets, f)
    else:
        with open(datasets_cache_path, 'rb') as f:
            datasets = pickle.load(f)

    model = Ensemble(args, tokenizer, target_size=num_authors).to(device)

    # load models
    lda_model_path = os.path.join('models', 'ckpt_mdl_lda_ep_10_hsize_768_dout_0.1')
    model.lda.load_state_dict(torch.load(lda_model_path)['model_state_dict'])
    gru_model_path = os.path.join('models', 'ckpt_gru_GRU_ep_50_hsize_768_dout_0.9.t49')
    model.gru.load_state_dict(torch.load(gru_model_path)['model_state_dict'])

    # run_eval(args, model, datasets, tokenizer,num_authors, split='validation')
    # run_eval(args, model, datasets, tokenizer,num_authors, split='test')
    ensemble_train(args, model, datasets, tokenizer)
    run_eval(args, model, datasets, tokenizer, num_authors,split='test')

  elif args.task == 'GRU':
     model = GRUNet(args, target_size=num_authors).to(device)
     run_eval(args, model, datasets, tokenizer, num_authors, split = 'validation')
     run_eval(args, model, datasets, tokenizer, num_authors, split='test')
     gru_train(args, model, datasets, tokenizer)
     run_eval(args, model, datasets, tokenizer, num_authors, split='test')
  else:
    raise ValueError(f'Invalid task: {args.task}')