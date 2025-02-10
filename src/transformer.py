import torch.nn as nn
from typing import Optional
import torch
from src.seed import set_seed
from math import log
import copy

set_seed()

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]

def clone_layer(layer, num_layers):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.norm = norm
        self.encoder_layers = clone_layer(encoder_layer, self.num_layers)

    def add_pos_embed(self, src, pos_embed):
        return src if pos_embed == None else (src + pos_embed)
    
    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos_embed: Optional[torch.Tensor] = None,
    ):
        encoder_output = self.add_pos_embed(src, pos_embed)
        for layer in self.encoder_layers:
            encoder_output = layer(src, src_mask, src_key_padding_mask)
        if self.norm != None:
            encoder_output = self.norm(encoder_output)
        return encoder_output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.num_layers = num_layers
        self.norm = norm
        self.decoder_layers = clone_layer(decoder_layer, self.num_layers)

    def add_pos_embed(self, tgt, pos_embed):
        return tgt if pos_embed == None else (tgt + pos_embed)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_pos_embed: Optional[torch.Tensor] = None,
    ):
        
        tgt = self.add_pos_embed(tgt, tgt_pos_embed)
        for layer in self.decoder_layers:
            decoder_output = layer(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
            )
        if self.norm != None:
            decoder_output = self.norm(decoder_output)
        return decoder_output


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: Optional[int] = 512,
        nhead: Optional[int] = 8,
        dim_feedforward: Optional[int] = 2048,
        dropout: Optional[float] = 0.2,
        encoder_num_layers: Optional[int] = 6,
        decoder_num_layers: Optional[int] = 6,
    ):
        super(Transformer, self).__init__()
        self.encoder_num_layers = encoder_num_layers
        self.decoder_num_layers = decoder_num_layers
        self.d_model = d_model
        if self.encoder_num_layers > 0:
            encoder_norm = None  # nn.LayerNorm(d_model)
            self.encoder = TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    norm_first=True,
                    batch_first=True,
                ),
                encoder_num_layers,
                encoder_norm,
            )
        if self.decoder_num_layers > 0:
            #self.query_embed = nn.Embedding(num_class, d_model)
            decoder_norm = None # nn.LayerNorm(d_model)
            self.decoder = TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    norm_first=True,
                    batch_first=True,
                ), 
                decoder_num_layers, 
                decoder_norm
            )

    def forward(
        self,
        src: torch.Tensor,
        tgt: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        pos_embed: Optional[torch.Tensor] = None,
        tgt_pos_embed: Optional[torch.Tensor] = None,
    ):
        encoder_embed = self.encoder(src, src_mask, src_key_padding_mask, pos_embed)
        if self.decoder_num_layers > 0:
            decoder_output = self.decoder(
                tgt,
                encoder_embed,
                src_mask,
                tgt_mask,
                src_key_padding_mask,
                tgt_key_padding_mask,
                pos_embed,
                tgt_pos_embed,
            )
            return decoder_output
        return encoder_embed


def build_transformer(args):
    return Transformer(
        d_model=args.embed_dim,
        nhead=args.num_heads,
        dim_feedforward=args.dim_forwardfeed,
        dropout=args.dropout,
        encoder_num_layers=args.encoder_num_layers,
        decoder_num_layers=args.decoder_num_layers,
    )