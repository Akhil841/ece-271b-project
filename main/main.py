import random
import itertools
import random
import itertools
import json
import numpy as np
import torch

from collections import defaultdict

import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm.notebook import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from transformers import BertTokenizer
from transformers import BertModel
from sklearn.model_selection import train_test_split

from model import SiameseBERT, contrastive_loss
from dataloader import load, create_training_data, prepare_training_data
from torch.utils.data import random_split


def eval_model(model,device, testing_data):
    
    # Visualize validation embeddings after the epoch
    model.eval()
    all_embeddings = []
    embedding_pairs = []
    all_authors = []
    labels = []
    val_indexs = range(len(testing_data["input_ids1"]))
    print(f"len of val indexs: {len(val_indexs)}")
    
    unique_authors = set(testing_data['author1'] + testing_data['author2'])
    print(unique_authors)
    
    with torch.no_grad():
        # Loop over indices in current validation split with progress bar
        for idx in tqdm(val_indexs, desc="Processing validation indices"):
            # Retrieve tokenized inputs for both texts (add batch dimension)
            sample_ids1 = testing_data["input_ids1"][idx].unsqueeze(0).to(device)
            sample_mask1 = testing_data["attention_mask1"][idx].unsqueeze(0).to(device)
            sample_ids2 = testing_data["input_ids2"][idx].unsqueeze(0).to(device)
            sample_mask2 = testing_data["attention_mask2"][idx].unsqueeze(0).to(device)
            
            # Get embeddings for both texts
            emb1, emb2 = model(sample_ids1, sample_mask1, sample_ids2, sample_mask2)
            all_embeddings.append(emb1.cpu())
            all_authors.append(testing_data['author1'][idx])  # original author for text1
            all_embeddings.append(emb2.cpu())
            all_authors.append(testing_data['author2'][idx])  # original author for text2
            
            embedding_pairs.append((emb1.cpu(), emb2.cpu()))
            labels.append(testing_data['labels'][idx])  # label for the pair
            
    

    # Split embedding_pairs and corresponding labels into training and test sets

    embeddings_np = torch.cat(all_embeddings, dim=0).numpy()

    # Convert and encode the author labels
    le = LabelEncoder()
    encoded_authors = le.fit_transform(np.array(all_authors))


    # --- SVM 1: Individual Embeddings ---
    X_train, X_test, y_train, y_test = train_test_split(embeddings_np, encoded_authors,
                                                        test_size=0.2, random_state=42)
    svm_ind = SVC(kernel='rbf')
    svm_ind.fit(X_train, y_train)
    preds_ind = svm_ind.predict(X_test)
    acc_ind = accuracy_score(y_test, preds_ind) * 100
    print(f"SVM (Individual embeddings) Accuracy: {acc_ind:.2f}%")
    
    # --- SVM 2: Pair-Difference SVM ---
    # Compute difference features per pair: take absolute difference.
    diffs = []
    for emb1, emb2 in embedding_pairs:
        diff = np.array([F.cosine_similarity(emb1, emb2).item()])
        diffs.append(diff)
    diffs = np.array(diffs)
    labels_arr = np.array(labels)
    
    X_train_pair, X_test_pair, y_train_pair, y_test_pair = train_test_split(diffs, labels_arr,
                                                                            test_size=0.2, random_state=42)
    svm_pair = SVC(kernel='rbf')
    svm_pair.fit(X_train_pair, y_train_pair)
    preds_pair = svm_pair.predict(X_test_pair)
    acc_pair = accuracy_score(y_test_pair, preds_pair) * 100
    print(f"SVM (Pair differences) Accuracy: {acc_pair:.2f}%")
    
    # --- SVM 3: Author Verification using Predictions from Individual Embeddings ---
    # For evaluation, split the embedding_pairs and corresponding pair labels.
    _, test_pairs, _, y_pair_test = train_test_split(embedding_pairs, labels,
                                                        test_size=0.2, random_state=42)
    # Method: predict the author for each text using the individual svm_ind
    # and decide the pair label as 1 (same) if predictions match, else 0.
    author_pred_labels = []
    for emb1, emb2 in test_pairs:
        pred1 = svm_ind.predict(emb1.numpy())
        pred2 = svm_ind.predict(emb2.numpy())
        author_pred_labels.append(1 if pred1[0] == pred2[0] else 0)
    acc_auth = accuracy_score(np.array(y_pair_test), np.array(author_pred_labels)) * 100
    print(f"SVM (Author verification by predicted labels) Accuracy: {acc_auth:.2f}%")
    
    # Optionally, you could also evaluate using the decision_function logits.
    author_pred_logits = []
    for emb1, emb2 in test_pairs:
        logits1 = svm_ind.decision_function(emb1.numpy())
        logits2 = svm_ind.decision_function(emb2.numpy())
        # Using Euclidean distance between logits and a threshold of 0.5
        distance = np.linalg.norm(logits1 - logits2)
        author_pred_logits.append(1 if distance < 0.5 else 0)
    acc_auth_logits = accuracy_score(np.array(y_pair_test), np.array(author_pred_logits)) * 100
    print(f"SVM (Author verification by logits threshold) Accuracy: {acc_auth_logits:.2f}%")
    

    # Concatenate embeddings into a (num_messages x hidden_size) numpy array
    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    # Reduce dimensions via PCA for visualization
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(all_embeddings)

    plt.figure(figsize=(8, 6))
    # Filter out authors labeled as 'other' (case-insensitive)
    filtered_authors = [a for a in all_authors if a.lower() != 'other']
    filtered_indices = [j for j, a in enumerate(all_authors) if a.lower() != 'other']
    # Use filtered embeddings corresponding to the filtered_authors
    embeddings_2d_filtered = embeddings_2d[filtered_indices]

    unique_authors = list(set(filtered_authors))
    colors = plt.cm.get_cmap("tab20", len(unique_authors))
    for i, author in enumerate(unique_authors):
        inds = [j for j, a in enumerate(filtered_authors) if a == author]
        plt.scatter(embeddings_2d_filtered[inds, 0], embeddings_2d_filtered[inds, 1],
                    color=colors(i), label=author, alpha=0.7)
    plt.title('Validation Embeddings')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()


if __name__ == "__main__":
    
    print("Loading data...")
    # Load the data
    data = load('data/reddit_comment_body_dec_2024.json')
    #print(type(data))
    print("Creating training data...")
    training_data = create_training_data(data, n_pairs=2000, n_authors=5)
    testing_data = create_training_data(data, n_pairs=100, n_authors=5)
    
    print("Preparing training data...")
    # Prepare the training data
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    training_data = prepare_training_data(training_data, tokenizer=tokenizer)
    testing_data = prepare_training_data(testing_data, tokenizer=tokenizer)

    debug = False
    
    # Split indices for training embeddings and SVM evaluation
    indices = list(range(len(training_data["labels"])))
    emb_indices, svm_indices = train_test_split(indices, test_size=0.2, random_state=42)

    # Create dictionaries for each split by indexing every field in training_data using list comprehensions
    training_embedding_data = {k: [training_data[k][i] for i in emb_indices] for k in training_data}
    training_svm_data = {k: [training_data[k][i] for i in svm_indices] for k in training_data}

    train_test_dataset = training_svm_data

    # Prepare dataset using tensors 
    training_dataset = TensorDataset(
        torch.tensor(training_embedding_data["input_ids1"]),
        torch.tensor(training_embedding_data["attention_mask1"]),
        torch.tensor(training_embedding_data["input_ids2"]),
        torch.tensor(training_embedding_data["attention_mask2"]),
        torch.tensor(training_embedding_data["labels"])
    )

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the updated SiameseBERT model and move it to device
    model = SiameseBERT().to(device)

    # Use AdamW optimizer with a low learning rate
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # Define training parameters
    n_epochs = 3
    batch_size = 8

    # Freeze all BERT parameters first
    for param in model.bert.parameters():
        param.requires_grad = False
    # Unfreeze the top 2 BERT layers for retraining
    for layer in model.bert.encoder.layer[-3:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Calculate total training steps for scheduler
    total_steps = (len(training_dataset) // batch_size) * n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=int(0.1 * total_steps), 
                                                num_training_steps=total_steps)

    train_subset = training_dataset
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

    # List to store average loss for each epoch
    all_epoch_losses = []

    print("Training the model...")
    # Train the model
    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0
        total = 0
        batch_losses = []  # record loss for each batch
        
        # Use enumerate to get the batch index in the loop
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}"), start=1):
            
            # Unpack batch and move to device
            input_ids1, mask1, input_ids2, mask2, labels = batch
            # Move tensors to device
            input_ids1 = input_ids1.to(device)
            mask1 = mask1.to(device)
            input_ids2 = input_ids2.to(device)
            mask2 = mask2.to(device)
            labels = labels.to(device).float()
            
            optimizer.zero_grad()
            emb1, emb2 = model(input_ids1, mask1, input_ids2, mask2)

            # Compute contrastive loss
            loss = contrastive_loss(emb1, emb2, labels, margin=1.0)
            
            loss.backward()
            optimizer.step()
            scheduler.step()  # update learning rate scheduler
            
            train_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            batch_losses.append(loss.item())
            
            if i == 1 or (i % max(1, int(0.2 * len(train_loader))) == 0):
                model.eval()
                eval_model(model, device, testing_data)
                eval_model(model, device, train_test_dataset)
                model.train()
        
        avg_loss = train_loss / total
        all_epoch_losses.append(avg_loss)
        print(f"Epoch {epoch} - Loss: {avg_loss:.4f} ")
        
        print(f"Epoch {epoch} Testing Results")
        eval_model(model, device, testing_data)
        
        print(f"Epoch {epoch} Training Results")
        eval_model(model, device, train_test_dataset)
        
        # Smooth out the loss graph using a moving average filter
        smoothing_window = 5  # Adjust the window size as needed
        smooth_losses = np.convolve(batch_losses, np.ones(smoothing_window)/smoothing_window, mode='valid')
        
        plt.figure(figsize=(8, 4))
        plt.plot(range(smoothing_window, len(batch_losses) + 1), smooth_losses, marker='o')
        plt.title(f'Epoch {epoch}: Smoothed Batch Loss')
        plt.xlabel('Batch Number')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()

    # Plot the average loss per epoch after all epochs finish
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, n_epochs+1), all_epoch_losses, marker='o')
    plt.title('Average Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.show()
