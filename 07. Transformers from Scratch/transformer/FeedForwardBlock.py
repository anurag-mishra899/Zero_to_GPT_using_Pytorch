import torch
import torch.nn as nn

class FeedForwardBlock(nn.Module):
    def __init__(self,d_m: int, d_ff: int,dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_m,d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff,d_m)

    def forward(self,x):
        ## x -> (batch,seq_len,d_m) -> (batch,seq_len,d_ff) -> (batch,seq_len,d_m)
        return self.linear_2(self.dropout(nn.ReLU(self.linear_1(x))))
