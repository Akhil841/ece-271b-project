
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from transformers import AdamW, BertConfig, BertModel, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup






class BERTToBiLSTM(nn.Module):
    def __init__(self, args, tokenizer, target_size):
        super().__init__()
        self.tokenizer = tokenizer
        # Initialize the Bert encoder
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.encoder.resize_token_embeddings(len(self.tokenizer))
        
        # BiLSTM: input size is BERT's hidden size (768)
        # Use provided args for LSTM hidden dimension and layers, or fallback defaults.
        self.lstm_hidden_dim = getattr(args, "lstm_hidden_dim", 256)
        self.num_layers = getattr(args, "lstm_layers", 2)
        self.bi_lstm = nn.LSTM(
            input_size=768,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Dropout layer and dense fully connected layer.
        self.dropout = nn.Dropout(args.drop_rate)
        # The fully connected layer implemented as a multi-layer perceptron (MLP).
        self.fc = nn.Sequential(
            nn.Linear(self.lstm_hidden_dim * 2, getattr(args, "mlp_hidden_dim", 128)),
            nn.ReLU(),
            nn.Dropout(getattr(args, "mlp_drop_rate", 0.1)),
            nn.Linear(getattr(args, "mlp_hidden_dim", 128), target_size)
        )

    def forward(self, inputs, targets):
        # Pass the inputs through the Bert encoder.
        bert_out = self.encoder(**inputs).last_hidden_state  # shape: (batch, seq_len, 768)
        
        # Pass the BERT output through the BiLSTM.
        lstm_out, _ = self.bi_lstm(bert_out)  # shape: (batch, seq_len, lstm_hidden_dim*2)
        
        # Aggregate LSTM outputs: concatenate the last forward and first backward hidden states.
        forward_last = lstm_out[:, -1, :self.lstm_hidden_dim]
        backward_first = lstm_out[:, 0, self.lstm_hidden_dim:]
        lstm_final = torch.cat((forward_last, backward_first), dim=1)
        
        # Apply dropout and the dense fully connected layer.
        lstm_final = self.dropout(lstm_final)
        logits = self.fc(lstm_final)
        
        return logits
    



class SiameseBERTToBiLSTM(nn.Module):
    def __init__(self, args, tokenizer, target_size):
        super().__init__()
        self.branch = BERTToBiLSTM(args, tokenizer, 128)
        self.classifier = nn.Sequential(
            nn.Sigmoid()
        )

    def forward(self, input_pair, targets):
        # Expect input_pair to be a tuple or list with two elements.
        left_input, right_input = input_pair

        # Process each input through the shared BERT-to-biLSTM branch.
        left_output = self.branch(left_input, targets)
        #print(f'left_output shape: {left_output.shape}')
        right_output = self.branch(right_input, targets)
        #print(f'right_output shape: {right_output.shape}')
        left_norm = F.normalize(left_output, p=2, dim=1)
        right_norm = F.normalize(right_output, p=2, dim=1)

        # Compute the cosine similarity between the outputs.
        diff = F.cosine_similarity(left_norm, right_norm, dim=1, eps=1e-6).unsqueeze(1)
        
        #print(f'diff shape: {diff.shape}')
        # Activate for binary classification.
        #score = self.classifier(diff)
        
        score = (diff + 1) / 2
        
            
        return score


