import torch
import torch.nn as nn
from transformer.Encoder import Encoder,EncoderBlock
from transformer.Decoder import Decoder,DecoderBlock
from transformer.Embeddings import PositionalEncoddings, InputEmbeddings
from transformer.ProjectionLayer import ProjectionLayer
from transformer.MultiHeadAttention import MultiHeadAttention
from transformer.FeedForwardBlock import FeedForwardBlock



class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embd: InputEmbeddings, 
                 tgt_embd: InputEmbeddings, src_pos: PositionalEncoddings, 
                 tgt_pos: PositionalEncoddings, proj_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embd = src_embd
        self.tgt_embd = tgt_embd
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.proj_layer = proj_layer

    def encode(self,src,src_mask):
        ## (batch,seq_len,d_m)
        src = self.src_embd(src)
        src = self.src_pos(src)
        return self.encoder(src,src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        ## (batch,seq_len,d_m)
        # (batch, seq_len, d_model)
        tgt = self.tgt_embd(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)    
    
    def project(self,x):
        return self.proj_layer(x)
    
def build_transformer(src_vocab_size,tgt_vocab_size,
                      src_seq_len,tgt_seq_len,
                      d_m = 512, N = 6, h = 8, dropout = 0.1,
                      d_ff = 2048):
    
    ## Create embedding layers
    src_embd = InputEmbeddings(d_m,src_vocab_size)
    tgt_embd = InputEmbeddings(d_m,tgt_vocab_size)

    ## Postional Layer
    src_pos = PositionalEncoddings(d_m,src_seq_len,dropout)
    tgt_pos = PositionalEncoddings(d_m,tgt_seq_len,dropout)

    ## create the encoder block
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_m,h,dropout)
        feed_forward_block = FeedForwardBlock(d_m,d_ff,dropout)
        encoder_block = EncoderBlock(d_m,encoder_self_attention_block,feed_forward_block,dropout)
        encoder_blocks.append(encoder_block)

    ## create the decoder block
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_m,h,dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_m,h,dropout)
        feed_forward_block = FeedForwardBlock(d_m,d_ff,dropout)
        decoder_block = DecoderBlock(d_m,decoder_self_attention_block,decoder_cross_attention_block,feed_forward_block,dropout)
        decoder_blocks.append(decoder_block)

    ## create the encoder and decoder
    encoder = Encoder(d_m,nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_m,nn.ModuleList(decoder_blocks))

    ## create the projection layer
    projection_layer = ProjectionLayer(d_m,tgt_vocab_size)

    ## create transformer Model
    transformer = Transformer(encoder,decoder,src_embd,tgt_embd,src_pos,tgt_pos,projection_layer)

    ## initialize paramters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer