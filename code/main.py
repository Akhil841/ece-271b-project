import os

import torch
from joblib import dump, load
from torch import nn
from tqdm import tqdm as progress_bar

from loss import ContrastiveLoss
from arguments import params
from dataloader import (
    check_cache,
    get_dataloader,
    prepare_features,
    prepare_inputs,
    process_data,
)
from load import load_data, load_tokenizer
from model import SiameseBERTToBiLSTM

from utils import check_directories, graph, set_seed, setup_gpus


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def baseline_train(args, model, datasets, tokenizer, num_authors):

    CHECKPOINT = 'ckpt_mdl_{}_ep_{}_hsize_{}_dout_{}'.format(args.task, args.n_epochs, args.hidden_dim, args.drop_rate)
    train_dataloader = get_dataloader(args, datasets['train'], 'train')

    model.optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    model.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model.optimizer, args.n_epochs)

    train_accuracies = []
    validation_accuracies = []

    # For Siamese networks with a Sigmoid classifier, use binary cross entropy loss.
    criterion = ContrastiveLoss()

    # Freeze all BERT layers except the top 2 layers for fine-tuning.
    num_top_layers = args.reinit_n_layers
    # Assuming the BERT encoder layers are in model.branch.encoder.encoder.layer
    total_layers = len(model.branch.encoder.encoder.layer)
    for i, layer in enumerate(model.branch.encoder.encoder.layer):
        if i < total_layers - num_top_layers:
            for param in layer.parameters():
                param.requires_grad = False
        else:
            for param in layer.parameters():
                param.requires_grad = True

    for epoch_count in range(args.n_epochs):
        losses = 0
        model.train()
        acc = 0
        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
            # Expect prepare_inputs to return a tuple of (left_input, right_input) and a binary label tensor.
            input_pair, labels = prepare_inputs(batch)
            # Ensure labels are float tensors with shape (batch, 1)
            labels = labels.float().unsqueeze(1)
            # Forward pass: model returns a probability between 0 and 1.
            score = model(input_pair, labels)
            loss = criterion(score, labels)
            loss.backward()

            # Compute predictions based on a 0.5 threshold.
            preds = (score >= 0.5).float()
            acc += (preds == labels).float().sum().item()

            model.optimizer.step()  # Update the weights.
            model.scheduler.step()  # Update the learning rate schedule.
            model.zero_grad()
            losses += loss.item()

        train_accuracies.append(acc / len(datasets['train']))
        validation_accuracies.append(run_eval(args, model, datasets, tokenizer, num_authors, split='validation'))
        print('training epoch', epoch_count, '| losses:', losses, '| accuracy:', acc / len(datasets['train']))

        if not os.path.isdir('models'):
            os.mkdir('models')

        # Save checkpoint.
        if (epoch_count % args.save_every == 0 and epoch_count != 0) or epoch_count == args.n_epochs - 1:
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
    # For Siamese networks, use contrastive loss
    criterion = ContrastiveLoss()

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for step, batch in progress_bar(enumerate(dataloader), total=len(dataloader)):
            # Assumption: prepare_inputs returns a tuple of (left_input, right_input) and binary labels
            input_pair, labels = prepare_inputs(batch)
            # Ensure labels are float tensors with shape (batch, 1)
            labels = labels.float().unsqueeze(1)
        
            # Assumption: model returns a similarity score between 0 and 1 after sigmoid activation
            scores = model(input_pair, labels)
            
            #print(f'Max score: {scores.max().item()}, Min score: {scores.min().item()}')
            
            #print(f'shape of csv: {scores.shape}')  
            loss += criterion(scores, labels).item()
            
            # Compute predictions based on a 0.5 threshold for binary classification
            preds = (scores >= 0.5).float()
            acc += (preds == labels).float().sum().item()

    # Calculate overall accuracy and average loss
    total_samples = len(datasets[split])
    avg_accuracy = acc / total_samples
    avg_loss = loss / len(dataloader)

    print(f'\n{split} acc: {avg_accuracy:.4f}, loss: {avg_loss:.4f}, dataset split {split} size: {total_samples}')
    return avg_accuracy


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
    model = SiameseBERTToBiLSTM(args, tokenizer, target_size=num_authors).to(device)
    
    #run_eval(args, model, datasets, tokenizer, num_authors, split='validation')
    #run_eval(args, model, datasets, tokenizer, num_authors, split='test')
    baseline_train(args, model, datasets, tokenizer, num_authors)
    run_eval(args, model, datasets, tokenizer, num_authors,num_authors, split='test')
  else:
    raise ValueError(f'Invalid task: {args.task}')