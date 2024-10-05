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
    def __init__(self,d_m: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        ## create a matrix of shape (seq_len,d_m)
        pe = torch.zeros(seq_len,d_m)
        ## create a vector of seq_len
        position = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(0) ## (seq_len,1)
        ## create a vector of d_m for divider
        div_term = torch.exp(torch.arange(0,d_m,2).float() * (-math.log(100000.0)/d_m)) ## (d_m/2)
        ## apply sine to even indices
        pe[:,0::2] = torch.sin(position*div_term)
        ## apply cosine to odd indices
        pe[:,1::2] = torch.cos(position*div_term)
        ## add a batch dimension to the pe
        pe = pe.unsqueeze(0) # (1,seq_len,d_m)
        ## register the positional encoding as buffer
        self.register_buffer('pe',pe)
    
    def forward(self,x):
        x = x + (self.pe[:,:x.shape[1],:]).requires_grad(False) ## (batch,seq_len,d_m)
        return x
