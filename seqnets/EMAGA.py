# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import List, Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
import math
from seqnets.incremental_decoding_utils import with_incremental_state
from seqnets.utils import get_activation_fn, relu2, laplace, softmax
from seqnets.EMA import MultiHeadEMA


logger = logging.getLogger(__name__)

__all__ = [
    'FairseqDropout', 'FairseqFeatureDropout',
    'SimpleRelativePositionalBias', 'RotaryRelativePositionalBias',
    'SequenceNorm', 
    'MovingAverageGatedAttention', 'NormalizedFeedForwardNetwork', 'GatedCrossAttention'

]

try:
    from apex.normalization import FusedLayerNorm as _FusedLayerNorm

    has_fused_layernorm = True

    class FusedLayerNorm(_FusedLayerNorm):
        @torch.jit.unused
        def forward(self, x):
            if not x.is_cuda:
                return super().forward(x)
            else:
                with torch.cuda.device(x.device):
                    return super().forward(x)

except ImportError:
    has_fused_layernorm = False


class FairseqDropout(nn.Module):

    def __init__(self, p, module_name=None):
        super().__init__()
        self.p = p
        self.module_name = module_name
        self.apply_during_inference = False

    def forward(self, x, batch_first: bool = False, inplace: bool = False):
        if self.training or self.apply_during_inference:
            return F.dropout(x, p=self.p, training=True, inplace=inplace)
        else:
            return x

    def make_generation_fast_(
        self,
        name: str,
        retain_dropout: bool = False,
        retain_dropout_modules: Optional[List[str]] = None,
        **kwargs
    ):
        if retain_dropout:
            if retain_dropout_modules is not None and self.module_name is None:
                logger.warning(
                    'Cannot enable dropout during inference for module {} '
                    'because module_name was not set'.format(name)
                )
            elif (
                retain_dropout_modules is None  # if None, apply to all modules
                or self.module_name in retain_dropout_modules
            ):
                logger.info(
                    'Enabling dropout during inference for module: {}'.format(name)
                )
                self.apply_during_inference = True
            else:
                logger.info('Disabling dropout for module: {}'.format(name))

    def extra_repr(self) -> str:
        return 'p={}'.format(self.p)


class FairseqFeatureDropout(nn.Module):

    def __init__(self, p, module_name=None):
        super().__init__()
        self.p = p
        self.module_name = module_name
        self.apply_during_inference = False

    def forward(self, x, batch_first: bool = False, inplace: bool = False):
        if self.training or self.apply_during_inference:
            if batch_first:
                # B x L x D -> B x D x L -> B x L x D
                return F.dropout2d(x.transpose(-1, -2), p=self.p, training=True, inplace=inplace).transpose(-1, -2)
            else:
                assert x.dim() == 3
                # L x B x D -> B x D x L -> L x B x D
                return F.dropout2d(x.permute(1, 2, 0), p=self.p, training=True, inplace=inplace).permute(2, 0, 1)
        else:
            return x

    def make_generation_fast_(
        self,
        name: str,
        retain_dropout: bool = False,
        retain_dropout_modules: Optional[List[str]] = None,
        **kwargs
    ):
        if retain_dropout:
            if retain_dropout_modules is not None and self.module_name is None:
                logger.warning(
                    'Cannot enable dropout during inference for module {} '
                    'because module_name was not set'.format(name)
                )
            elif (
                retain_dropout_modules is None  # if None, apply to all modules
                or self.module_name in retain_dropout_modules
            ):
                logger.info(
                    'Enabling dropout during inference for module: {}'.format(name)
                )
                self.apply_during_inference = True
            else:
                logger.info('Disabling dropout for module: {}'.format(name))

    def extra_repr(self) -> str:
        return 'p={}'.format(self.p)

class SimpleRelativePositionalBias(nn.Module):

    def __init__(self, max_positions):
        super().__init__()
        self.max_positions = max_positions
        self.rel_pos_bias = nn.Parameter(torch.Tensor(2 * max_positions - 1))
        self.reset_parameters()

    def reset_parameters(self):
        std = 0.02
        nn.init.normal_(self.rel_pos_bias, mean=0.0, std=std)

    def forward(self, seq_len):
        if seq_len > self.max_positions:
            raise ValueError('Sequence length {} going beyond max length {}'.format(seq_len, self.max_positions))

        # seq_len * 2 -1
        b = self.rel_pos_bias[(self.max_positions - seq_len):(self.max_positions + seq_len - 1)]
        # seq_len * 3 - 1
        t = F.pad(b, (0, seq_len))
        # (seq_len * 3 - 1) * seq_len
        t = torch.tile(t, (seq_len,))
        t = t[:-seq_len]
        # seq_len x (3 * seq_len - 2)
        t = t.view(seq_len, 3 * seq_len - 2)
        r = (2 * seq_len - 1) // 2
        start = r
        end = t.size(1) - r
        t = t[:, start:end]
        return t

    def extra_repr(self) -> str:
        return 'max positions={}'.format(self.max_positions)

class RotaryRelativePositionalBias(nn.Module):
    def __init__(self, embed_dim, max_positions):
        super().__init__()
        assert embed_dim % 2 == 0
        self.embed_dim = embed_dim
        self.max_positions = max_positions
        self.sine, self.cosine = RotaryRelativePositionalBias.get_sinusoid_embeddings(max_positions, embed_dim)
        self.alpha = nn.Parameter(torch.Tensor(1, embed_dim))
        self.beta = nn.Parameter(torch.Tensor(1, embed_dim))
        self.register_buffer("_float_tensor", torch.FloatTensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        std = 0.02
        nn.init.normal_(self.alpha, mean=0.0, std=std)
        nn.init.normal_(self.beta, mean=0.0, std=std)

    @staticmethod
    def get_sinusoid_embeddings(max_positions: int, embedding_dim: int):
        half_dim = embedding_dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(max_positions, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        return torch.sin(emb), torch.cos(emb)

    def rotary(self, x):
        n, d = x.size()
        x1, x2 = torch.chunk(x, 2, dim=-1)
        if self.sine is None or n > self.sine.size(0):
            self.sine, self.cosine = RotaryRelativePositionalBias.get_sinusoid_embeddings(n, d)
            self.max_positions = n
        self.sine = self.sine.to(self._float_tensor)
        self.cosine = self.cosine.to(self._float_tensor)

        sin = self.sine[:n]
        cos = self.cosine[:n]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=1)

    def forward(self, seq_len):
        a = self.rotary(self.alpha.expand(seq_len, self.embed_dim))
        b = self.rotary(self.beta.expand(seq_len, self.embed_dim))
        t = torch.einsum('mk,nk->mn', a, b)
        return t

    def extra_repr(self) -> str:
        return 'dim={}, max positions={}'.format(self.embed_dim, self.max_positions)
    

def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    if torch.jit.is_scripting():
        export = True
    if not export and torch.cuda.is_available() and has_fused_layernorm:
        return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)

class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, affine=True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.affine = affine
        if affine:
            self.scalar = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('scalar', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.constant_(self.scalar, 1.0)

    def forward(self, x):
        mean_square = torch.mean(torch.square(x), dim=self.dim, keepdim=True)
        if self.scalar is not None:
            x = self.scalar * x

        x = x * torch.rsqrt(mean_square + self.eps)
        return x

    def extra_repr(self) -> str:
        return 'dim={dim}, eps={eps}, affine={affine}'.format(**self.__dict__)


class RMSNorm(nn.Module):
    def __init__(self, number_features, eps=1e-6, affine=True):
        super().__init__()
        self.num_features = number_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.Tensor(self.num_features))
        else:
            self.register_parameter('weight', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.constant_(self.weight, 1.0)

    def forward(self, x):
        mean_square = torch.mean(torch.square(x), dim=-1, keepdim=True)
        if self.weight is not None:
            x = x * self.weight

        x = x * torch.rsqrt(mean_square + self.eps)
        return x

    def extra_repr(self) -> str:
        return '{num_features}, eps={eps}, affine={affine}'.format(**self.__dict__)


class SequenceNorm(nn.Module):
    def __init__(self, norm_type, embedding_dim, eps=1e-5, affine=True, export=False):
        super().__init__()
        if norm_type == 'layernorm':
            self.norm = LayerNorm(embedding_dim, eps=eps, elementwise_affine=affine, export=export)
        elif norm_type == 'scalenorm':
            self.norm = ScaleNorm(dim=-1, eps=eps, affine=affine)
        elif norm_type == 'rmsnorm':
            self.norm = RMSNorm(embedding_dim, eps=eps, affine=affine)
        elif norm_type == 'batchnorm':
            self.norm = nn.BatchNorm1d(embedding_dim, eps=eps, affine=affine)
        elif norm_type == 'syncbatchnorm':
            self.norm = nn.SyncBatchNorm(embedding_dim, eps=eps, affine=affine)
        else:
            raise ValueError('Unknown norm type: {}'.format(norm_type))

    def normalize(self, x):
        if isinstance(self.norm, nn.modules.batchnorm._BatchNorm):
            assert x.dim() == 3
            x = x.permute(1, 2, 0)
            x = self.norm(x)
            return x.permute(2, 0, 1)
        else:
            return self.norm(x)

    def forward(self, x):
        return self.normalize(x)
    
@with_incremental_state
class MovingAverageGatedAttention(nn.Module):

    def __init__(
        self,
        embed_dim, # 输入x的形状为 [seq_len, bs, embed_dim]
        zdim, # attention dim, 对应(7）中的z
        hdim, # out_dim of attention，对应（10）中的v
        ndim, # 对应原文中的h
        dropout=0.0,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        activation='silu',
        attention_activation='softmax',
        bidirectional=False,
        chunk_size=-1,
        truncation=None,
        norm_type='layernorm',
        prenorm=True,
        norm_affine=True,
        feature_dropout=False,
        rel_pos_bias='simple',
        max_positions=1024,
        export=False,
        with_move = 1, # 1表示都有，0表示没有move，2表示没有attention
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.hdim = hdim
        self.zdim = zdim
        self.ndim = ndim
        self.activation = get_activation_fn(activation=activation)
        self.attention_activation = attention_activation
        self.scaling = self.zdim ** -0.5 if attention_activation == 'softmax' else None

        dropout_module = FairseqFeatureDropout if feature_dropout else FairseqDropout
        self.dropout = dropout_module(dropout, module_name=self.__class__.__name__)
        self.hidden_dropout = dropout_module(hidden_dropout, module_name=self.__class__.__name__)
        # Attention dropout is standard dropout
        self.attention_dropout = FairseqDropout(attention_dropout, module_name=self.__class__.__name__)

        self.chunk_size = chunk_size
        self.prenorm = prenorm
        self.norm = SequenceNorm(norm_type, embed_dim, affine=norm_affine, export=export)

        self.move = MultiHeadEMA(embed_dim, ndim=ndim, bidirectional=bidirectional, truncation=truncation)
        self.with_move = with_move

        self.v_proj = nn.Linear(embed_dim, hdim)
        self.mx_proj = nn.Linear(embed_dim, zdim + hdim + 2 * embed_dim)
        self.h_proj = nn.Linear(hdim, embed_dim)

        self.gamma = Parameter(torch.Tensor(2, zdim))
        self.beta = Parameter(torch.Tensor(2, zdim))

        self.max_positions = max_positions
        max_positions = max_positions if chunk_size < 0 else chunk_size
        if rel_pos_bias == 'simple':
            self.rel_pos_bias = SimpleRelativePositionalBias(max_positions)
        elif rel_pos_bias == 'rotary':
            self.rel_pos_bias = RotaryRelativePositionalBias(zdim, max_positions)
        else:
            raise ValueError('unknown relative position bias: {}'.format(rel_pos_bias))

        self.reset_parameters()

        self.onnx_trace = False
        self.tpu = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

    def reset_parameters(self):
        std = 0.02
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=std)
        nn.init.constant_(self.v_proj.bias, 0.0)

        nn.init.normal_(self.mx_proj.weight, mean=0.0, std=std)
        nn.init.constant_(self.mx_proj.bias, 0.0)

        nn.init.normal_(self.h_proj.weight, mean=0.0, std=std)
        nn.init.constant_(self.h_proj.bias, 0.0)

        nn.init.normal_(self.gamma, mean=0.0, std=std)
        nn.init.constant_(self.beta, 0.0)

    def element_attention(self, q, k, padding_mask, attn_mask, before_attn_fn):
        slen = k.size(2)
        if padding_mask is not None:
            # B x K x C
            inverse_mask = 1.0 - padding_mask.type_as(q)
            # B x K x 1
            lengths = inverse_mask.sum(dim=-1, keepdim=True)
            # B x K x 1 x 1
            lengths = lengths.clamp(min=1.0).unsqueeze(-1)
        else:
            lengths = slen
            inverse_mask = None

        if attn_mask is not None:
            # C x 1
            lengths = attn_mask.sum(dim=-1, keepdim=True)

        # C x C
        bias = self.rel_pos_bias(slen)
        if slen != q.size(2):
            assert q.size(2) == 1
            # 1 x C
            bias = bias[-1:]

        # B x K x C x C
        qk = torch.matmul(q, k.transpose(2, 3)) / lengths + bias

        if before_attn_fn:
            return qk

        if self.attention_activation == 'relu2':
            attn_weights = relu2(qk).type_as(qk)
        elif self.attention_activation == 'laplace':
            attn_weights = laplace(qk).type_as(qk)
        else:
            raise ValueError('Unknown attention activation function: {}'.format(self.attention_activation))

        if inverse_mask is not None:
            attn_weights = attn_weights * inverse_mask.unsqueeze(2)

        if attn_mask is not None:
            attn_weights = attn_weights * attn_mask

        return attn_weights

    def softmax_attention(self, q, k, padding_mask, attn_mask, before_attn_fn):
        slen = k.size(2)
        # C x C
        bias = self.rel_pos_bias(slen)
        if slen != q.size(2):
            assert q.size(2) == 1
            # 1 x C
            bias = bias[-1:]

        # scaled attention
        q = q * self.scaling
        # B x K x C x C
        qk = torch.matmul(q, k.transpose(2, 3)) + bias

        if attn_mask is not None:
            qk = qk + attn_mask

        if padding_mask is not None:
            padding_mask_all = padding_mask.all(dim=-1, keepdim=True)
            padding_mask = torch.logical_and(padding_mask, ~padding_mask_all)
            qk = qk.masked_fill(padding_mask.unsqueeze(2).to(torch.bool), float('-inf'))

        if before_attn_fn:
            return qk

        attn_weights = softmax(qk, dim=-1, onnx_trace=self.onnx_trace).type_as(qk)
        return attn_weights

    def forward(
        self,
        x,
        padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_attn_fn: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_attn_fn (bool, optional): return the raw attention
                weights and values before the attention softmax.
        """

        seq_len, bsz, embed_dim = x.size()
        assert embed_dim == self.embed_dim

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
        else:
            saved_state = None

        residual = x
        if self.prenorm:
            x = self.norm(x)

        # L x B x E
        v = self.activation(self.v_proj(x))

        # L x B x D
        if self.with_move == 0:
            mx = x
        else:
            mx = self.move(x, padding_mask, incremental_state)
        mx = self.dropout(mx)

        if self.with_move == 2:
            return mx, None
        else:
            # L x B x D -> L x B x (2*D+S+E)
            base = self.mx_proj(mx)
            u, zr, hx = torch.split(base, [self.embed_dim, self.zdim + self.hdim, self.embed_dim], dim=-1)
            # L x B x D
            u = torch.sigmoid(u) # (13)
            # L x B x (E+S)
            z, r = torch.split(F.silu(zr), [self.zdim, self.hdim], dim=-1) # （7） （12）
            # L x B x S -> L x B x 1 x S -> L x B x 2 x S
            z = z.unsqueeze(2) * self.gamma + self.beta
            # L x B x 2 x S -> L x B x S
            q, k = torch.unbind(z, dim=2)

            # L x B x D -> B x L x D
            q = q.transpose(0, 1) # （8）
            k = k.transpose(0, 1) # （9）
            v = v.transpose(0, 1) # （10）

            if saved_state is not None:
                # assert self.chunk_size < 0 or q.size(1) <= self.chunk_size
                # saved states are stored with shape (bsz, seq_len, dim)
                if "prev_key" in saved_state:
                    prev_key = saved_state["prev_key"]
                    assert prev_key is not None
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
                if "prev_value" in saved_state:
                    prev_value = saved_state["prev_value"]
                    assert prev_value is not None
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
                prev_padding_mask: Optional[Tensor] = None
                if "prev_padding_mask" in saved_state:
                    prev_padding_mask = saved_state["prev_padding_mask"]
                padding_mask = MovingAverageGatedAttention._append_prev_padding_mask(
                    padding_mask=padding_mask,
                    prev_padding_mask=prev_padding_mask,
                    batch_size=bsz,
                    seq_len=k.size(1),
                )

                if self.chunk_size < 0:
                    saved_state["prev_key"] = k
                    saved_state["prev_value"] = v
                    saved_state["prev_key_padding_mask"] = padding_mask
                else:
                    curr_len = k.size(1) % self.chunk_size
                    if curr_len == 0:
                        if "prev_key" in saved_state:
                            del saved_state["prev_key"]
                            del saved_state["prev_value"]
                            del saved_state["prev_key_padding_mask"]
                    else:
                        saved_state["prev_key"] = k
                        saved_state["prev_value"] = v
                        saved_state["prev_key_padding_mask"] = padding_mask
                # In this branch incremental_state is never None
                assert incremental_state is not None
                self._set_input_buffer(incremental_state, saved_state)

            ctx_len = k.size(1)
            if self.chunk_size < 0:
                # B x L x S -> B x 1 x L x S
                q = q.unsqueeze(1)
                k = k.unsqueeze(1)
                v = v.unsqueeze(1)
                if padding_mask is not None:
                    # B x L -> B x 1 x L
                    padding_mask = padding_mask.unsqueeze(1)
            else:
                if seq_len < self.chunk_size:
                    q = q.unsqueeze(1)
                else:
                    # B x L x S -> B x K x C x S
                    nc = seq_len // self.chunk_size
                    q = q.reshape(bsz, nc, self.chunk_size, self.zdim)

                if ctx_len < self.chunk_size:
                    k = k.unsqueeze(1)
                    v = v.unsqueeze(1)
                    if padding_mask is not None:
                        padding_mask = padding_mask.unsqueeze(1)
                else:
                    # B x L x S -> B x K x C x S
                    nc = ctx_len // self.chunk_size
                    k = k.reshape(bsz, nc, self.chunk_size, self.zdim)
                    v = v.reshape(bsz, nc, self.chunk_size, self.hdim)
                    if padding_mask is not None:
                        # B x L -> B x K x C
                        padding_mask = padding_mask.view(bsz, nc, self.chunk_size)

            # This is part of a workaround to get around fork/join parallelism
            # not supporting Optional types.
            if padding_mask is not None and padding_mask.dim() == 0:
                padding_mask = None

            if self.attention_activation == 'softmax':
                attn_weights = self.softmax_attention(q, k, padding_mask, attn_mask, before_attn_fn)
            else:
                attn_weights = self.element_attention(q, k, padding_mask, attn_mask, before_attn_fn)

            if before_attn_fn:
                return attn_weights, v

            v = self.hidden_dropout(v, batch_first=True)
            kernel = self.attention_dropout(attn_weights)
            # B x K x C x E -> B x L x E -> L x B x E
            h = torch.matmul(kernel, v).view(bsz, seq_len, self.hdim).transpose(0, 1)
            # L x B x E -> L x B x D
            h = self.activation(hx + self.h_proj(h * r)) # （14）
            h = self.dropout(h)
            # L x B x D
            out = torch.addcmul(residual, u, h - residual) # （15）

            if not self.prenorm:
                out = self.norm(out)

            if need_weights:
                return out, attn_weights
            else:
                return out, None

    def _get_input_buffer(self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], buffer: Dict[str, Optional[Tensor]]):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    @torch.jit.export
    def reorder_incremental_state(
            self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], new_order: Tensor
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    @staticmethod
    def _append_prev_padding_mask(
        padding_mask: Optional[Tensor],
        prev_padding_mask: Optional[Tensor],
        batch_size: int,
        seq_len: int,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_padding_mask is not None and padding_mask is not None:
            new_padding_mask = torch.cat([prev_padding_mask, padding_mask], dim=1)
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_padding_mask is not None:
            filler = torch.zeros((batch_size, seq_len - prev_padding_mask.size(1)), device=prev_padding_mask.device)
            new_padding_mask = torch.cat([prev_padding_mask, filler.bool()], dim=1)
        elif padding_mask is not None:
            filler = torch.zeros((batch_size, seq_len - padding_mask.size(1)), device=padding_mask.device)
            new_padding_mask = torch.cat([filler.bool(), padding_mask], dim=1)
        else:
            new_padding_mask = prev_padding_mask
        return new_padding_mask

    def extra_repr(self) -> str:
        return 'edim={}, zdim={}, hdim={}, ndim={}, chunk={}, attn_act={}, prenorm={}'.format(self.embed_dim, self.zdim,
                                                                                  self.hdim, self.ndim, self.chunk_size,
                                                                                  self.attention_activation, self.prenorm)
    

@with_incremental_state
class GatedCrossAttention(nn.Module):
    """Gated Structured State Attention.

    See "" for more details.
    """

    def __init__(
        self,
        embed_dim,
        zdim,
        ndim=2,
        dropout=0.0,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        activation='silu',
        attention_activation='softmax',
        norm_type='layernorm',
        prenorm=True,
        norm_affine=True,
        feature_dropout=False,
        rel_pos_bias='simple',
        max_positions=1024,
        export=False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.zdim = zdim
        self.ndim = ndim
        self.activation = get_activation_fn(activation=activation)
        self.attention_activation = attention_activation
        self.scaling = self.zdim ** -0.5 if attention_activation == 'softmax' else None

        dropout_module = FairseqFeatureDropout if feature_dropout else FairseqDropout
        self.dropout = dropout_module(dropout, module_name=self.__class__.__name__)
        self.hidden_dropout = dropout_module(hidden_dropout, module_name=self.__class__.__name__)
        # Attention dropout is standard dropout
        self.attention_dropout = FairseqDropout(attention_dropout, module_name=self.__class__.__name__)

        self.prenorm = prenorm
        self.norm = SequenceNorm(norm_type, embed_dim, affine=norm_affine, export=export)

        self.k_proj = nn.Linear(embed_dim, zdim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, 2 * embed_dim + zdim)
        self.h_proj = nn.Linear(embed_dim, embed_dim)

        self.max_positions = max_positions
        if rel_pos_bias == 'simple':
            self.rel_pos_bias = SimpleRelativePositionalBias(max_positions)
        elif rel_pos_bias == 'rotary':
            self.rel_pos_bias = RotaryRelativePositionalBias(zdim, max_positions)
        else:
            raise ValueError('unknown relative position bias: {}'.format(rel_pos_bias))

        self.reset_parameters()

        self.onnx_trace = False
        self.tpu = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

    def reset_parameters(self):
        std = 0.02
        nn.init.normal_(self.k_proj.weight, mean=0.0, std=std)
        nn.init.constant_(self.k_proj.bias, 0.0)

        nn.init.normal_(self.v_proj.weight, mean=0.0, std=std)
        nn.init.constant_(self.v_proj.bias, 0.0)

        nn.init.normal_(self.q_proj.weight, mean=0.0, std=std)
        nn.init.constant_(self.q_proj.bias, 0.0)

        nn.init.normal_(self.h_proj.weight, mean=0.0, std=std)
        nn.init.constant_(self.h_proj.bias, 0.0)

    def element_attention(self, q, k, key_padding_mask, pidx, before_attn_fn):
        bsz, clen, _ = k.size()
        slen = q.size(1) if pidx is None else pidx + 1
        if key_padding_mask is not None:
            # B x L1
            inverse_mask = 1.0 - key_padding_mask.type_as(q)
            # B x 1 x 1
            lengths = inverse_mask.sum(dim=-1).view(bsz, 1, 1)
        else:
            lengths = clen
            inverse_mask = None

        # L x L1
        bias = self.rel_pos_bias(max(slen, clen))[:, :clen]
        if pidx is not None:
            assert q.size(1) == 1
            # L1
            bias = bias[pidx]
        else:
            # L2 x L1
            bias = bias[:slen]

        # B x L2 x L1
        qk = torch.bmm(q, k.transpose(1, 2)) / lengths + bias

        if before_attn_fn:
            return qk

        if self.attention_activation == 'relu2':
            attn_weights = relu2(qk).type_as(qk)
        elif self.attention_activation == 'laplace':
            attn_weights = laplace(qk).type_as(qk)
        else:
            raise ValueError('Unknown attention activation function: {}'.format(self.attention_activation))

        if inverse_mask is not None:
            attn_weights = attn_weights * inverse_mask.unsqueeze(1)

        return attn_weights

    def softmax_attention(self, q, k, key_padding_mask, pidx, before_attn_fn):
        bsz, clen, _ = k.size()
        slen = q.size(1) if pidx is None else pidx + 1

        # L x L1
        bias = self.rel_pos_bias(max(slen, clen))[:, :clen]
        if pidx is not None:
            assert q.size(1) == 1
            # L1
            bias = bias[pidx]
        else:
            # L2 x L1
            bias = bias[:slen]

        # scaled attention
        q = q * self.scaling
        # B x L2 x L1
        qk = torch.bmm(q, k.transpose(1, 2)) + bias

        if key_padding_mask is not None:
            qk = qk.masked_fill(key_padding_mask.unsqueeze(1).to(torch.bool), float('-inf'))

        if before_attn_fn:
            return qk

        attn_weights = softmax(qk, dim=-1, onnx_trace=self.onnx_trace).type_as(qk)
        return attn_weights

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        padding_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = False,
        static_kv: bool = False,
        before_attn_fn: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            padding_mask (ByteTensor, optional): mask to exclude
                queries that are pads, of shape `(batch, tgt_len)`, where
                padding elements are indicated by 1s.
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            static_kv (bool, optional): static key and value pair.
            before_attn_fn (bool, optional): return the raw attention
                weights and values before the attention softmax.
        """

        seq_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            pidx = 0
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                assert static_kv
                key = value = None
        else:
            pidx = None
            saved_state = None

        q = query
        if self.prenorm:
            q = self.norm(q)

        # L2 x B x (2*D+S)
        base = self.q_proj(q)
        u, r, q = torch.split(base, [self.embed_dim, self.embed_dim, self.zdim], dim=-1)

        # L2 x B x D
        u = torch.sigmoid(u)
        r = F.silu(r)

        if key is None:
            assert value is None
            k = v = None
        else:
            # L1 x B x S
            k = self.k_proj(key)
            v = self.activation(self.v_proj(key))

        # L2 x B x S -> B x L2 x S
        q = q.transpose(0, 1)
        if k is not None:
            k = k.transpose(0, 1)
        if v is not None:
            v = v.transpose(0, 1)

        if saved_state is not None:
            # saved states are stored with shape (bsz, seq_len, dim)
            if "prev_key" in saved_state:
                prev_key = saved_state["prev_key"]
                assert prev_key is not None
                k = prev_key
            if "prev_value" in saved_state:
                prev_value = saved_state["prev_value"]
                assert prev_value is not None
                v = prev_value
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
                key_padding_mask = prev_key_padding_mask
            if "prev_num_steps" in saved_state:
                _prev_num_steps = saved_state["prev_num_steps"]
                pidx = _prev_num_steps + 1

            saved_state["prev_key"] = k
            saved_state["prev_value"] = v
            saved_state["prev_key_padding_mask"] = key_padding_mask
            saved_state["prev_num_steps"] = pidx
            # In this branch incremental_state is never None
            assert incremental_state is not None
            self._set_input_buffer(incremental_state, saved_state)

        ctx_len = k.size(1)
        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == ctx_len

        if self.attention_activation == 'softmax':
            attn_weights = self.softmax_attention(q, k, key_padding_mask, pidx, before_attn_fn)
        else:
            attn_weights = self.element_attention(q, k, key_padding_mask, pidx, before_attn_fn)

        if before_attn_fn:
            return attn_weights, v

        v = self.hidden_dropout(v, batch_first=True)
        kernel = self.attention_dropout(attn_weights)
        # B x L2 x D -> L2 x B x D
        h = torch.bmm(kernel, v).transpose(0, 1)
        # L2 x B x D
        h = self.activation(self.h_proj(h * r))
        h = self.dropout(h)
        out = torch.addcmul(query, u, h - query)

        if not self.prenorm:
            out = self.norm(out)

        if need_weights:
            return out, attn_weights
        else:
            return out, None

    def _get_input_buffer(self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], buffer: Dict[str, Optional[Tensor]]):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    @torch.jit.export
    def reorder_incremental_state(
            self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], new_order: Tensor
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None and isinstance(input_buffer_k, Tensor):
                    if input_buffer_k.size(0) == new_order.size(0):
                        break
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def extra_repr(self) -> str:
        return 'edim={}, zdim={}, ndim={}, attn_act={}, prenorm={}'.format(self.embed_dim, self.zdim, self.ndim,
                                                                           self.attention_activation, self.prenorm)


class NormalizedFeedForwardNetwork(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_hidden_dim,
        dropout=0.0,
        hidden_dropout=0.0,
        activation='silu',
        norm_type='layernorm',
        prenorm=True,
        norm_affine=True,
        feature_dropout=False,
        export=False,
    ):
        super().__init__()

        self.embedding_dim = embed_dim
        self.hidden_dim = ffn_hidden_dim
        self.act_fn = activation
        self.activation = get_activation_fn(activation)

        dropout_module = FairseqFeatureDropout if feature_dropout else FairseqDropout
        self.dropout = dropout_module(dropout, module_name=self.__class__.__name__)
        self.hidden_dropout = dropout_module(hidden_dropout, module_name=self.__class__.__name__)

        self.prenorm = prenorm
        self.norm = SequenceNorm(norm_type, embed_dim, affine=norm_affine, export=export)

        self.fc1 = nn.Linear(embed_dim, ffn_hidden_dim)
        self.fc2 = nn.Linear(ffn_hidden_dim, embed_dim)

        self.reset_parameters()

    def reset_parameters(self):
        std = 0.02
        nn.init.normal_(self.fc1.weight, mean=0.0, std=std)
        nn.init.constant_(self.fc1.bias, 0.0)

        nn.init.normal_(self.fc2.weight, mean=0.0, std=std)
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, x):
        residual = x

        if self.prenorm:
            x = self.norm(x)

        x = self.activation(self.fc1(x))
        x = self.hidden_dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = x + residual

        if not self.prenorm:
            x = self.norm(x)

        return x

    def extra_repr(self) -> str:
        return 'edim={}, hdim={}, act={}, prenorm={}'.format(self.embedding_dim, self.hidden_dim, self.act_fn, self.prenorm)
