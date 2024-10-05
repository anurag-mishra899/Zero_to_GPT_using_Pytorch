import torch 
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self,d_m: int, vocab_size: int) -> None:
        super().__init__()
        self.d_m = d_m
        self.embeddings = nn.Embedding(vocab_size,d_m)

    def forward(self,x):
        ## (batch, seq_len) -> (batch,seq_len,d_m)
        ## Multiply by sqrt(d_m) to scale the embeddings
        return self.embeddings(x) * math.sqrt(self.d_m)
    

class PositionalEncoddings(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)
    
    def forward(self,x):
    #     x = x + (self.pe[:,:x.shape[1],:]).requires_grad(False) ## (batch,seq_len,d_m)
    #     return x
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)
