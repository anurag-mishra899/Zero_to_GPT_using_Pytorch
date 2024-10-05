import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_m: int, heads : int, dropout: float) -> None:
        super().__init__()
        self.d_m = d_m
        self.heads = self.heads
        self.d_k = self.d_m/self.heads
        self.dropout = nn.Dropout(dropout)
        self.w_Q = nn.Linear(d_m,d_m, bias=False) ## Wq
        self.w_K = nn.Linear(d_m,d_m, bias=False) ## Wk
        self.w_V = nn.Linear(d_m,d_m, bias=False) ## Wv
        self.w_O = nn.Linear(d_m,d_m, bias=False) ## Wo

    @staticmethod
    def attention(query,key,value,mask,dropout: nn.Dropout):
        d_k = query.shape[-1]
        ## Just apply the formula from the paper
        ## (batch,heads,seq_len,d_k) -> (batch,heads,seq_len,seq_len)
        attention_score = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            attention_score.masked_fill_(mask==0,-1e9)
        attention_score = attention_score.softmax(dim=1) ## (batch,heads,seq_len,seq_len)
        if dropout is not None:
            attention_score = dropout(attention_score)
        ## (batch,heads,seq_len,seq_len) -> (batch,heads,seq_len,d_k)
        return (attention_score @ value), attention_score

    def forward(self,q,k,v,mask):
        query = self.w_Q(q)
        key = self.w_K(k)
        value = self.w_V(v)

        # (batch,seq_len,d_m) -> (batch,seq_len,heads,d_k) -> (batch.head,seq_len,d_k)
        query = query.view(query.shape[0],query.shape[1],self.heads,self.d_k).transpose(1,2)
        key = key.view(key.shape[0],key.shape[1],self.heads,self.d_k).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1],self.heads,self.d_k).transpose(1,2)

        x,self.attention_score = MultiHeadAttention.attention(query,key,value,mask,self.dropout)

        # combine all aheads together
        # (batch,heads,seq_len,d_k) -> (batch,seq_len,d_k)
        x = x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h*self.d_k)

        return self.w_O(x)
        