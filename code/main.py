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
from sklearn.decomposition import PCA
from tqdm import tqdm
import os


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

            # Compute predictions based on a 0.5 threshold.
            preds = (score >= 0.5).float()
            #print(f'preds shape: {preds.shape}')
            #print(f'first 50 preds: {preds[:50]}')
            #print(f"first 50 labels: {labels[:50]}")
            acc += (preds == labels).float().sum().item()
            #print(f'acc: {acc}')

            model.zero_grad()       # Reset gradients BEFORE new batch
            loss.backward()
            
            
            losses += loss.item()
            

        model.optimizer.step() 
        model.scheduler.step()  # Update the learning rate schedule.

        train_accuracies.append(acc / len(datasets['train']))
        validation_accuracies.append(run_eval(args, model, datasets, tokenizer, num_authors, split='validation'))
        print('training epoch', epoch_count, '| losses:', losses, '| accuracy:', acc / len(datasets['train']))
        
        visualize_embeddings(args, model, datasets)
        
    
        if not os.path.isdir('results'):
            os.mkdir('results')
        with open(os.path.join('results', 'accuracy.txt'), 'a') as f:
            f.write(f"Epoch {epoch_count}: train_accuracy = {train_accuracies[-1]:.4f}, validation_accuracy = {validation_accuracies[-1]:.4f}\n")

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


def visualize_embeddings(args, model, datasets):
    import matplotlib.pyplot as plt

    model.eval()
    dataloader = get_dataloader(args, datasets['validation'], 'validation')
    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for _, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            # Prepare inputs with classes=True returns a tuple of two lists of string labels.
            input_pair, labels = prepare_inputs(batch, classes=True)
            # Use only one side (e.g., left input) for the embedding visualization.
            left_input, _ = input_pair  
            # Compute embeddings from the left branch.
            emb = model.branch(left_input)
            embeddings_list.append(emb.cpu())
            # Collect left side labels (list of strings) directly.
            # Instead of appending tensors, extend a Python list.
            labels_list.extend(labels[0])
    
    all_embeddings = torch.cat(embeddings_list, dim=0).numpy()
    
    # Map each unique string label to an integer for coloring.
    unique_labels = sorted(set(labels_list))
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    all_label_int = [label_to_int[label] for label in labels_list]
    
    # Reduce dimensions to 2 for visualization.
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(all_embeddings)
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1],
        c=all_label_int, cmap='viridis', alpha=0.5
    )
    plt.colorbar(scatter, ticks=range(len(unique_labels)), label='Label index')
    plt.clim(-0.5, len(unique_labels)-0.5)
    plt.title('Embeddings Visualization')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    
    # Save plot to results folder.
    if not os.path.isdir('results'):
        os.mkdir('results')
    print('Saving visualization to results folder')
    save_path = os.path.join('results', 'embedding_visualization.png')
    plt.savefig(save_path)
    plt.close()

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
  if args.task == 'dl-contrastive':
    model = SiameseBERTToBiLSTM(args, tokenizer, target_size=num_authors).to(device)
    
    #run_eval(args, model, datasets, tokenizer, num_authors, split='validation')
    #run_eval(args, model, datasets, tokenizer, num_authors, split='test')
    baseline_train(args, model, datasets, tokenizer, num_authors)
    run_eval(args, model, datasets, tokenizer, num_authors,num_authors, split='test')
  else:
    raise ValueError(f'Invalid task: {args.task}')