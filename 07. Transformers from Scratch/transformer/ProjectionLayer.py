import torch 
import torch.nn as nn


class ProjectionLayer(nn.Module):
    def __init__(self, d_m,vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_m,vocab_size)

    def forward(self,x):
        return self.proj(x)