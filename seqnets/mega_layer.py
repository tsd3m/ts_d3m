import torch
import torch.nn as nn
from seqnets.EMAGA import MovingAverageGatedAttention, GatedCrossAttention, NormalizedFeedForwardNetwork
from typing import Optional, Dict
from torch import Tensor

class MegaEncoderLayer(nn.Module):
    """Encoder layer block.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, emb_dim, zdim, hdim, ndim,
                 encoder_ffn_embed_dim,
                 dropout=0.0, 
                 attention_dropout=0.0,
                 hidden_dropout=0.0,
                 activation='silu',
                 attention_activation='softmax',
                 bidirectional=True, # 这里要选True比较好
                 chunk_size=-1,
                 truncation=None,
                 norm_type='layernorm',
                 prenorm=True,
                 norm_affine=True,
                 feature_dropout=False,
                 rel_pos_bias='simple',
                 max_positions=1024,
                 with_move=1,
                 ):
        super().__init__()

        self.embed_dim = emb_dim
        self.zdim = zdim
        self.hdim = hdim
        self.ndim = ndim
        self.encoder_ffn_embed_dim = encoder_ffn_embed_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.activation = activation
        self.attention_dropout = attention_dropout
        self.attention_activation = attention_activation
        self.bidirectional = bidirectional
        self.chunk_size = chunk_size
        self.truncation = truncation
        self.norm_type = norm_type
        self.prenorm = prenorm
        self.norm_affine = norm_affine
        self.feature_dropout = feature_dropout
        self.rel_pos_bias = rel_pos_bias
        self.max_positions = max_positions
        self.with_move = with_move


        self.mega_layer = self.build_mega_layer(self.embed_dim)
        if self.encoder_ffn_embed_dim > 0:
            self.nffn = self.build_nffn_layer(self.embed_dim)
        else:
            self.nffn = None

    def build_mega_layer(self, embed_dim):
        return MovingAverageGatedAttention(
            embed_dim=embed_dim,
            zdim=self.zdim,
            hdim=self.hdim,
            ndim=self.ndim,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            hidden_dropout=self.hidden_dropout,
            chunk_size=self.chunk_size,
            truncation=self.truncation,
            rel_pos_bias=self.rel_pos_bias,
            max_positions=self.max_positions,
            activation=self.activation,
            attention_activation=self.attention_activation,
            bidirectional=self.bidirectional,
            norm_type=self.norm_type,
            prenorm=self.prenorm,
            feature_dropout=self.feature_dropout,
            with_move=self.with_move
        )

    def build_nffn_layer(self, embed_dim):
        return NormalizedFeedForwardNetwork(
            embed_dim=embed_dim,
            ffn_hidden_dim=self.encoder_ffn_embed_dim,
            dropout=self.dropout,
            hidden_dropout=self.hidden_dropout,
            activation=self.activation,
            norm_type=self.norm_type,
            prenorm=self.prenorm,
            feature_dropout=self.feature_dropout,
        )

    def forward(self, x, encoder_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        x, _ = self.mega_layer(x, encoder_padding_mask)
        if self.nffn is not None:
            x = self.nffn(x)

        return x