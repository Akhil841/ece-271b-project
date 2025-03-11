from transformers import BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseBERT(nn.Module):
    def __init__(self, pretrained_model_name="bert-base-uncased", hidden_size=768, dropout_prob=0.1, n_layers=4):
        """
        Initializes the SiameseBERT network.
        
        Args:
            pretrained_model_name (str): Name of the pretrained BERT model.
            hidden_size (int): Hidden size of the BERT model.
            dropout_prob (float): Dropout probability.
            n_layers (int): Number of last layers to use for layer-wise representations.
        """
        super(SiameseBERT, self).__init__()
        self.n_layers = n_layers
        
        # Load BERT and enable output of hidden states for layer-wise representations
        self.bert = BertModel.from_pretrained(pretrained_model_name, output_hidden_states=True)
        self.dropout = nn.Dropout(dropout_prob)
        
        # Learnable weights for combining the last n_layers representations
        self.layer_weights = nn.Parameter(torch.ones(n_layers) / n_layers)
        # Learnable vector for attention pooling across tokens to capture grammatical structure
        self.attn_vector = nn.Parameter(torch.randn(hidden_size))
        
        # Fully connected block: main branch
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU()
        )
        # Residual branch to map the original embedding to the reduced dimension
        self.fc_residual = nn.Linear(hidden_size, hidden_size // 2)
        
        # Layer normalization after adding the residual connection
        self.layer_norm = nn.LayerNorm(hidden_size // 2)

    def attention_pool(self, tokens, mask):
        """
        Applies attention pooling over token embeddings to capture grammatical structure.
        
        Args:
            tokens (torch.Tensor): Token embeddings of shape (batch_size, seq_len, hidden_size).
            mask (torch.Tensor): Attention mask of shape (batch_size, seq_len).
            
        Returns:
            torch.Tensor: Pooled embedding of shape (batch_size, hidden_size).
        """
        # Compute attention scores using a learnable vector
        scores = torch.matmul(tokens, self.attn_vector)  # shape: (batch_size, seq_len)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=1).unsqueeze(-1)  # shape: (batch_size, seq_len, 1)
        pooled = torch.sum(tokens * attn_weights, dim=1)  # shape: (batch_size, hidden_size)
        return pooled

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        """
        Forward pass for the Siamese network.
        
        Args:
            input_ids1 (torch.Tensor): Input IDs for the first text, shape (batch_size, seq_len).
            attention_mask1 (torch.Tensor): Attention mask for the first text.
            input_ids2 (torch.Tensor): Input IDs for the second text.
            attention_mask2 (torch.Tensor): Attention mask for the second text.
            
        Returns:
            prob (torch.Tensor): Predicted probability (same author) of shape (batch_size, 1).
            embed1 (torch.Tensor): Refined embedding for the first text.
            embed2 (torch.Tensor): Refined embedding for the second text.
        """
        # Batch processing: concatenate inputs along the batch dimension
        input_ids = torch.cat([input_ids1, input_ids2], dim=0)  # Shape: (2*batch_size, seq_len)
        attention_mask = torch.cat([attention_mask1, attention_mask2], dim=0)
        
        # Get BERT outputs with hidden states
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states  # Tuple of layers: (layer0, layer1, ..., layer_n)
        
        # Instead of just using the [CLS] token, apply attention pooling on each layer
        pooled_embeddings = []
        for i in range(1, self.n_layers + 1):
            tokens = hidden_states[-i]  # shape: (batch, seq_len, hidden_size)
            pooled = self.attention_pool(tokens, attention_mask)
            pooled_embeddings.append(pooled)
        weighted_pooled = sum(w * emb for w, emb in zip(self.layer_weights, pooled_embeddings))
        weighted_pooled = self.dropout(weighted_pooled)
        
        # Fully connected transformation with residual connection and normalization
        refined = self.fc_layers(weighted_pooled) + self.fc_residual(weighted_pooled)
        refined = self.layer_norm(refined)
        
        # Split back into two halves for the two inputs
        batch_size = input_ids1.size(0)
        embed1 = refined[:batch_size]
        embed2 = refined[batch_size:]
        
        return embed1, embed2

def contrastive_loss(embedding1, embedding2, label, margin=0.5):
    """
    Computes the contrastive loss for metric learning.
    
    Args:
        embedding1 (torch.Tensor): Embeddings for the first input.
        embedding2 (torch.Tensor): Embeddings for the second input.
        label (torch.Tensor): 1 if same author, 0 otherwise.
        margin (float): Margin for the contrastive loss.
    
    Returns:
        torch.Tensor: Mean contrastive loss.
    """
     # Normalize the embeddings to unit vectors
    output1_norm = F.normalize(embedding1, p=2, dim=1)
    output2_norm = F.normalize(embedding2, p=2, dim=1)
    
    # Compute cosine similarity between each pair in the batch
    cosine_sim = F.cosine_similarity(output1_norm, output2_norm, dim=1)
    
    # For similar pairs (label==1), we want the similarity to be close to 1.
    # Loss for similar pairs: 1 - cosine similarity.
    loss_similar = (1 - cosine_sim) * label
    
    # For dissimilar pairs (label==0), if the cosine similarity is greater than the margin,
    # we want to push it down. Otherwise, no loss is incurred.
    loss_dissimilar = F.relu(cosine_sim - margin) * (1 - label)
    
    # Combine the losses and take the mean over the batch.
    loss = torch.mean(loss_similar + loss_dissimilar)
    return loss

