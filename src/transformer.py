import torch.nn as nn
from typing import Optional
import torch
from mamba_ssm import Mamba2
from math import log
import copy


class MambaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MambaRMSNorm is equivalent to T5LayerNorm and LlamaRMSNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{self.weight.shape[0]}, eps={self.variance_epsilon}"


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
            if src_mask != None or src_key_padding_mask != None:
                encoder_output = layer(encoder_output, src_mask, src_key_padding_mask)
            else:
                encoder_output = layer(encoder_output)
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
            tgt = layer(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
            )
        if self.norm != None:
            tgt = self.norm(tgt)
        return tgt


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: Optional[int] = 512,
        nhead: Optional[int] = 8,
        dim_feedforward: Optional[int] = 2048,
        dropout: Optional[float] = 0.2,
        encoder_type: Optional[str] = "mamba",
        encoder_num_layers: Optional[int] = 6,
        decoder_num_layers: Optional[int] = 6,
        use_pos_encoding: Optional[bool] = True,
        max_seq_length: Optional[int] = 100,
    ):
        super(Transformer, self).__init__()
        self.encoder_num_layers = encoder_num_layers
        self.decoder_num_layers = decoder_num_layers
        self.d_model = d_model
        self.use_pos_encoding = use_pos_encoding
        self.encoder_type = encoder_type
        self.max_seq_length = max_seq_length
        if self.use_pos_encoding:
            self.pos_encoder = PositionalEncoding(
                d_model, max_seq_length=self.max_seq_length
            )
        if self.encoder_num_layers > 0:
            if encoder_type == "mamba":
                encoder_norm = None  # MambaRMSNorm(d_model)  # nn.RMSNorm(d_model)  # nn.LayerNorm(d_model)
                self.encoder = TransformerEncoder(
                    Mamba2(
                        d_model=d_model,
                        d_state=64,
                        d_conv=4,  # Local convolution width
                        expand=2,  # Block expansion factor
                    ),
                    encoder_num_layers,
                    encoder_norm,
                )
            else:
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
            decoder_norm = None  # nn.LayerNorm(d_model)

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
                decoder_norm,
            )

    def forward(
        self,
        src: torch.Tensor,
        tgt: Optional[torch.Tensor] = None,  # Query embeddings
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_pos_embed: Optional[torch.Tensor] = None,
    ):
        if self.use_pos_encoding:
            src = self.pos_encoder(src)
        if self.encoder_type == "mamba":
            encoder_embed = self.encoder(src)
        else:
            encoder_embed = self.encoder(src, src_mask, src_key_padding_mask)
        if self.decoder_num_layers > 0:
            decoder_output = self.decoder(
                tgt,
                encoder_embed,
                src_mask,
                tgt_mask,
                src_key_padding_mask,
                tgt_key_padding_mask,
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
        encoder_type=args.encoder_type,
        encoder_num_layers=args.encoder_num_layers,
        decoder_num_layers=args.decoder_num_layers,
        use_pos_encoding=args.use_pos_encoding,
        max_seq_length=args.max_seq_length,
    )
