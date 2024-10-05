import torch
import torch.nn as nn
from transformer.LayerNormalization import LayerNormalization

class ResidualConnect(nn.Module):
    def __init__(self,features: int, dropout: float):
        super().__init__()
        self.norm = LayerNormalization(features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,x,sublayer):
        ## x -> (batch,seq_len,d_m)
        x = x + self.dropout(sublayer(self.norm(x)))
        return x 