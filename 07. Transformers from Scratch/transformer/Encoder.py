import torch 
import torch.nn as nn
from transformer.MultiHeadAttention import MultiHeadAttention
from transformer.FeedForwardBlock import FeedForwardBlock
from transformer.ResidualConnect import ResidualConnect
from transformer.LayerNormalization import LayerNormalization


class EncoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttention, feedforward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feedforward_block
        self.residual_connection = nn.ModuleList([ResidualConnect(features,dropout) for _ in range(2)])

    def forward(self,x, src_mask):
        x = self.residual_connection[0](x,lambda x: self.self_attention_block(x,x,x,src_mask))
        x = self.residual_connection[1](x,self.feed_forward_block)
        return x
    

class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)
    
    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)
