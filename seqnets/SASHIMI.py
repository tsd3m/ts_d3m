import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from functools import partial
from seqnets.utils import LinearActivation, Activation
from seqnets.EMA import MultiHeadEMA
from seqnets.mega_layer import MegaEncoderLayer


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1):
        super(Conv, self).__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              dilation=dilation, 
                              padding=self.padding,
                             stride=stride)
        
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        out = self.conv(x)
        return out

class DownPool(nn.Module):
    def __init__(self, d_input, expand, pool):
        super().__init__()
        self.d_output = d_input * expand
        self.pool = pool

        self.linear = LinearActivation(
            d_input * pool,
            self.d_output,
            transposed=True,
        )

    def forward(self, x):
        x = rearrange(x, '... h (l s) -> ... (h s) l', s=self.pool)
        x = self.linear(x)
        return x

    def step(self, x, state, **kwargs):
        """
        x: (..., H)
        """

        if x is None: return None, state
        state.append(x)
        if len(state) == self.pool:
            x = rearrange(torch.stack(state, dim=-1), '... h s -> ... (h s)')
            x = x.unsqueeze(-1)
            x = self.linear(x)
            x = x.squeeze(-1)
            return x, []
        else:
            return None, state

    def default_state(self, *args, **kwargs):
        return []


class UpPool(nn.Module):
    def __init__(self, d_input, expand, pool, causal=True):
        super().__init__()
        self.d_output = d_input // expand
        self.pool = pool
        self.causal = causal

        self.linear = LinearActivation(
            d_input,
            self.d_output * pool,
            transposed=True,
        )

    def forward(self, x, skip=None):
        x = self.linear(x)

        if self.causal:
            x = F.pad(x[..., :-1], (1, 0)) # Shift to ensure causality
        x = rearrange(x, '... (h s) l -> ... h (l s)', s=self.pool)

        if skip is not None:
            x = x + skip
        return x

    def step(self, x, state, **kwargs):
        """
        x: (..., H)
        """
        assert len(state) > 0
        y, state = state[0], state[1:]
        if len(state) == 0:
            assert x is not None
            x = x.unsqueeze(-1)
            x = self.linear(x)
            x = x.squeeze(-1)
            x = rearrange(x, '... (h s) -> ... h s', s=self.pool)
            state = list(torch.unbind(x, dim=-1))
        else: assert x is None
        return y, state

    def default_state(self, *batch_shape, device=None):
        state = torch.zeros(batch_shape + (self.d_output, self.pool), device=device) # (batch, h, s)
        state = list(torch.unbind(state, dim=-1)) # List of (..., H)
        return state
    

class ResidualBlock(nn.Module):

    def __init__(
        self, 
        d_model, 
        seq_len,
        layer,
        dropout,
        diffusion_step_embed_dim_out,
        cond_in_channels,
        stride,
        direction=1
    ):
        
        """
        Residual mega block.

        Args:
            d_model: dimension of the model
            bidirectional: use bidirectional S4 layer
            glu: use gated linear unit in the S4 layer
            dropout: dropout rate
            direction: 
                1: feature oriented attention
                0: seq_len oriented attention
        """
        super().__init__()

        self.layer = layer
        self.direction = direction
        self.norm = nn.LayerNorm(d_model if direction==1 else seq_len)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

        self.fc_t = nn.Linear(diffusion_step_embed_dim_out, d_model if direction==1 else seq_len)
        self.cond_conv = Conv(cond_in_channels, d_model, kernel_size=stride, stride=stride)
        # self.fc_label = nn.Linear(label_embed_dim, d_model)  if label_embed_dim is not None else None
        
        
    def forward(self, x, cond, diffusion_step_embed):
        """
        Input x is shape (B, K, L)
        cond: (bs, channel, K, L)
        diff_emb: (bs, emb_dim)
        """
        B, _, K, S = cond.shape
        if self.direction == 0:
            x = x.transpose(1, 2).contiguous()
            # cond = cond.reshape(B, -1, K)
        # elif self.direction == 1:
        cond = cond.reshape(B, -1, S)

        B, K, L = x.shape
        
        # import pdb; pdb.set_trace()
        # add in diffusion step embedding
        part_t = self.fc_t(diffusion_step_embed).unsqueeze(2)
        z = x + part_t
        
        
        
        # Prenorm
        z = self.norm(z.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        # import pdb; pdb.set_trace()
        z = self.layer(z.permute(2, 0, 1).contiguous(), None) # (l, B, K)
        # import pdb; pdb.set_trace()
        cond = self.cond_conv(cond) # (B, K, L)
        #cond = self.fc_label(cond)
        
        # import pdb; pdb.set_trace()
        if self.direction == 0:
            z = z.permute(2, 1, 0).contiguous() + cond.permute(2, 0, 1).contiguous()
        elif self.direction == 1:
            z = z + cond.permute(2, 0, 1).contiguous() # (L, B, K)
            
        # Dropout on the output of the layer
        z = self.dropout(z)

        # import pdb; pdb.set_trace()
        # Residual connection
        if self.direction == 1:
            x = z.permute(1, 2, 0).contiguous() + x
        elif self.direction == 0:
            x = z.permute(1, 2, 0).contiguous() + x.permute(0, 2, 1).contiguous()

        return x

class GaussianFourierProjection(nn.Module):
  """Gaussian Fourier embeddings for noise levels."""

  def __init__(self, embedding_size=256, scale=1.0):
    super().__init__()
    self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
  

class DiffusionGFEmbedding(nn.Module):
    def __init__(self, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        assert embedding_dim % 2 == 0
        self.embedding = GaussianFourierProjection(embedding_size=embedding_dim//2)
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        # import pdb; pdb.set_trace()
        x = self.embedding(diffusion_step)
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    

class Sashimi(nn.Module):
    def __init__(self, config, 
                 feat_dim,
                 cond_in_channels, 
                 cond_channels, # 给到residual layer中的channel
                 n_layers=6,
                 pool=[2, 2], 
                 expand=2):
        super().__init__()
        self.config = config
        self.feat_dim = H = feat_dim
        ori_feat_dim = feat_dim // 2
        self.cond_channels = cond_channels
        self.diff_md_config = config['model']
        def mega_block(emb_dim, cond_channels, stride):
            # 输入输出的形状应该是 [seq_len, bs, emb_dim]
            layer = MegaEncoderLayer(emb_dim=emb_dim,
                                     zdim=self.diff_md_config['zdim'],
                                     hdim=self.diff_md_config['hdim'],
                                     ndim=self.diff_md_config['ndim'],
                                     encoder_ffn_embed_dim=self.diff_md_config['ffn_embed_dim'],
                                     dropout=self.diff_md_config['dropout'],
                                     attention_dropout=self.diff_md_config['att_dropout'],
                                     bidirectional=bool(self.diff_md_config['bidirectional']),
                                     chunk_size=self.diff_md_config['chunk_size'],
                                     truncation=None,
                                     norm_type=self.diff_md_config['norm_type'],
                                     prenorm=bool(self.diff_md_config['prenorm']),
                                     norm_affine=bool(self.diff_md_config['norm_affine']),
                                     feature_dropout=bool(self.diff_md_config['feat_dropout']),
                                     rel_pos_bias=self.diff_md_config['rel_pos_bias'],
                                     max_positions=self.diff_md_config['max_positions'])
            return ResidualBlock(
                d_model=emb_dim,
                layer=layer,
                dropout=self.diff_md_config['dropout'],
                diffusion_step_embed_dim_out=self.config['diffusion']['diffusion_embedding_dim'],
                cond_in_channels=cond_channels,
                stride=stride
            )
        
        self.diffusion_embedding = DiffusionGFEmbedding(
            embedding_dim=self.config['diffusion']['diffusion_embedding_dim']
        )

        self.cond_layer = nn.Sequential(
            nn.Conv2d(cond_in_channels, cond_channels // ori_feat_dim, 1),
            
        )
    
        d_layers = []
        for i, p in enumerate(pool):
            for _ in range(n_layers):
                if i == 0:
                    d_layers.append(mega_block(H, cond_channels, 1))
                elif i == 1:
                    d_layers.append(mega_block(H, cond_channels, p))
            d_layers.append(DownPool(H, expand, p))
            H *= expand
        
        c_layers = []
        for _ in range(n_layers):
            c_layers.append(mega_block(H, cond_channels, pool[1]*2))


        u_layers = []
        for i, p in enumerate(pool[::-1]):
            block = []
            H //= expand
            block.append(UpPool(H * expand, expand, p, 
                                causal=not bool(self.diff_md_config['bidirectional'])))
        
            for _ in range(n_layers):
                if i == 0:
                    block.append(mega_block(H, cond_channels, pool[0]))
                elif i == 1:
                    block.append(mega_block(H, cond_channels, 1))
            
            u_layers.append(nn.ModuleList(block))
        

        self.d_layers = nn.ModuleList(d_layers)
        self.c_layers = nn.ModuleList(c_layers)
        self.u_layers = nn.ModuleList(u_layers)
        self.norm = nn.LayerNorm(H)

        self.end_conv = nn.Sequential(
            nn.Conv1d(feat_dim, feat_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(feat_dim, ori_feat_dim, kernel_size=1)
        )   

    def forward(self, x, diffusion_steps, cond):
        '''
        x: [bs, embd_dim, K, L]
        cond: [bs, cond_dim, K, L]
        diffusion_steps: [bs,]
        '''

        B, channel, K, L = x.shape
        x = x.reshape(B, channel*K, L) # 输入的维度
        cond = self.cond_layer(cond)
        cond = cond.reshape(B, -1, L)
        diffusion_emb = self.diffusion_embedding(diffusion_steps)
        # import pdb; pdb.set_trace()

        outputs = []
        outputs.append(x)
        for layer in self.d_layers:
            if isinstance(layer, ResidualBlock):
                x = layer(x, cond, diffusion_emb)
            else:
                x = layer(x)

            outputs.append(x)
        # print(x.shape, [i.shape for i in outputs])

        for layer in self.c_layers:
            if isinstance(layer, ResidualBlock):
                x = layer(x, cond, diffusion_emb)
            else:
                x = layer(x)

        x = x + outputs.pop()
        # print(x.shape)


        for block in self.u_layers:
            for layer in block:
                # print(x.shape)
                if isinstance(layer, ResidualBlock):
                    x = layer(x, cond, diffusion_emb)
                else:
                    x = layer(x)
                x = x + outputs.pop()
        
        x = self.end_conv(x)
        return x


class D3M_Sashimi(nn.Module):
    def __init__(self, config, 
                 feat_dim,
                 seq_len,
                 cond_in_channels, 
                 cond_channels, # 给到residual layer中的channel
                 n_layers=6,
                 pool=[2, 2], 
                 expand=2):
        super().__init__()
        self.config = config

        self.feat_dim = H = feat_dim
        self.seq_len = S = seq_len

        ori_feat_dim = feat_dim // 2
        self.cond_channels = cond_channels
        self.diff_md_config = config['model']
        def mega_block(emb_dim, seq_len, cond_channels, stride, direction=1):

            layer = MegaEncoderLayer(emb_dim=emb_dim if direction==1 else seq_len,
                                     zdim=self.diff_md_config['zdim'],
                                     hdim=self.diff_md_config['hdim'],
                                     ndim=self.diff_md_config['ndim'],
                                     encoder_ffn_embed_dim=self.diff_md_config['ffn_embed_dim'],
                                     dropout=self.diff_md_config['dropout'],
                                     attention_dropout=self.diff_md_config['att_dropout'],
                                     bidirectional=bool(self.diff_md_config['bidirectional']),
                                     chunk_size=self.diff_md_config['chunk_size'],
                                     truncation=None,
                                     norm_type=self.diff_md_config['norm_type'],
                                     prenorm=bool(self.diff_md_config['prenorm']),
                                     norm_affine=bool(self.diff_md_config['norm_affine']),
                                     feature_dropout=bool(self.diff_md_config['feat_dropout']),
                                     rel_pos_bias=self.diff_md_config['rel_pos_bias'],
                                     max_positions=self.diff_md_config['max_positions'])
            return ResidualBlock(
                d_model=emb_dim,
                seq_len=seq_len,
                layer=layer,
                dropout=self.diff_md_config['dropout'],
                diffusion_step_embed_dim_out=self.config['diffusion']['diffusion_embedding_dim'],
                cond_in_channels=cond_channels,
                stride=stride,
                direction=direction
            )
        
        self.diffusion_embedding = DiffusionGFEmbedding(
            embedding_dim=self.config['diffusion']['diffusion_embedding_dim']
        )

        self.cond_layer = nn.Sequential(
            nn.Conv2d(cond_in_channels, cond_channels // ori_feat_dim, 1),
            
        )
    
        d_layers = []
        for i, p in enumerate(pool):
            for k in range(n_layers):
                direction = 1 - k%2
                if i == 0:
                    d_layers.append(mega_block(H, S, cond_channels, 1, direction=direction))
                elif i == 1:
                    d_layers.append(mega_block(H, S, cond_channels, p, direction=direction))
            d_layers.append(DownPool(H, expand, p))
            H *= expand
            S //= expand
        
        c_layers = []
        for k in range(n_layers):
            direction = 1 - k%2
            c_layers.append(mega_block(H, S, cond_channels, pool[1]*2, direction=direction))

        H1 = H2 = H
        S1 = S2 = S

        u_layers1 = []
        for i, p in enumerate(pool[::-1]):
            block = []
            H1 //= expand
            S1 *= expand
            block.append(UpPool(H1 * expand, expand, p, 
                                causal=not bool(self.diff_md_config['bidirectional'])))
        
            for k in range(n_layers):
                direction = 1 - k%2
                if i == 0:
                    block.append(mega_block(H1, S1, cond_channels, pool[0], direction=direction))
                elif i == 1:
                    block.append(mega_block(H1, S1, cond_channels, 1, direction=direction))
            
            u_layers1.append(nn.ModuleList(block))

        
        u_layers2 = []
        for i, p in enumerate(pool[::-1]):
            block = []
            H2 //= expand
            S2 *= expand
            block.append(UpPool(H2 * expand, expand, p, 
                                causal=not bool(self.diff_md_config['bidirectional'])))
        
            for k in range(n_layers):
                direction = 1 - k%2
                if i == 0:
                    block.append(mega_block(H2, S2, cond_channels, pool[0], direction=direction))
                elif i == 1:
                    block.append(mega_block(H2, S2, cond_channels, 1, direction=direction))
            
            u_layers2.append(nn.ModuleList(block))
        

        self.d_layers = nn.ModuleList(d_layers)
        self.c_layers = nn.ModuleList(c_layers)
        self.u_layers1 = nn.ModuleList(u_layers1)
        self.u_layers2 = nn.ModuleList(u_layers2)
        self.norm = nn.LayerNorm(H)

        self.end_conv1 = nn.Conv1d(feat_dim, ori_feat_dim, kernel_size=1)

        self.end_conv2 = nn.Conv1d(feat_dim, ori_feat_dim, kernel_size=1)

        nn.init.zeros_(self.end_conv1.weight)
        nn.init.zeros_(self.end_conv2.weight)

    def forward(self, x, diffusion_steps, cond):
        '''
        x: [bs, embd_dim, K, L]
        cond: [bs, cond_dim, K, L]
        diffusion_steps: [bs,]
        '''

        B, channel, K, L = x.shape
        x = x.reshape(B, channel*K, L) # 输入的维度
        cond = self.cond_layer(cond)
        # cond = cond.reshape(B, -1, L)
        diffusion_emb = self.diffusion_embedding(diffusion_steps)
        # import pdb; pdb.set_trace()

        outputs = []
        outputs.append(x)
        # import pdb; pdb.set_trace()
        for layer in self.d_layers:
            if isinstance(layer, ResidualBlock):
                x = layer(x, cond, diffusion_emb)
            else:
                x = layer(x)

            outputs.append(x)

        for layer in self.c_layers:
            if isinstance(layer, ResidualBlock):
                x = layer(x, cond, diffusion_emb)
            else:
                x = layer(x)

        num = 1
        x1 = x2 =  x + outputs[-num]

        
        for block in self.u_layers1:
            for layer in block:
                num += 1
                if isinstance(layer, ResidualBlock):
                    x1 = layer(x1, cond, diffusion_emb)
                else:
                    x1 = layer(x1)
                x1 = x1 + outputs[-num]
        
        x1 = self.end_conv1(x1)

        num = 1
        for block in self.u_layers2:
            for layer in block:
                num += 1
                if isinstance(layer, ResidualBlock):
                    x2 = layer(x2, cond, diffusion_emb)
                else:
                    x2 = layer(x2)
                x2 = x2 + outputs[-num]
        
        x2 = self.end_conv2(x2)

        return (x1, ), x2