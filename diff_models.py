import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from seqnets.mega_layer import MegaEncoderLayer


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

def get_mega_block(emb_dim, zdim, hdim, ndim, encoder_ffn_embed_dim,
                   dropout=0.,
                   attention_dropout=0.,
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
                    with_move=1):
    encoder_layer = MegaEncoderLayer(emb_dim, zdim, hdim, ndim, encoder_ffn_embed_dim,
                                     dropout, attention_dropout, hidden_dropout, activation,
                                     attention_activation, bidirectional, chunk_size, truncation,
                                     norm_type, prenorm, norm_affine, feature_dropout, rel_pos_bias,
                                     max_positions, with_move=with_move)
    return encoder_layer


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class GaussianFourierProjection(nn.Module):
  """Gaussian Fourier embeddings for noise levels."""

  def __init__(self, embedding_size=256, scale=1.0):
    super().__init__()
    self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table

# diffusion step embedding with gaussian fourier embedding
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

    # def _build_embedding(self, num_steps, dim=64):
    #     steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
    #     frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
    #     table = steps * frequencies  # (T,dim)
    #     table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
    #     return table

class diff_CSDI(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, cond_info, diffusion_step):
        B, inputdim, K, L = x.shape

        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        y = y + cond_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip
    

class EMA_ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, 
                 t_emb_dim, f_emb_dim, zdim, hdim, ndim, encoder_ffn_embed_dim,
                 dropout=0., attention_dropout=0., hidden_dropout=0., activation='silu',
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
                 time_with_move=1,
                 feat_with_move=1):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = get_mega_block(emb_dim=t_emb_dim, zdim=zdim, hdim=hdim, ndim=ndim,
                                         encoder_ffn_embed_dim=encoder_ffn_embed_dim,
                                         dropout=dropout, attention_dropout=attention_dropout,
                                         hidden_dropout=hidden_dropout, activation=activation,
                                         attention_activation=attention_activation, 
                                         bidirectional=bidirectional, chunk_size=chunk_size,
                                         truncation=truncation, norm_type=norm_type, prenorm=prenorm, 
                                         norm_affine=norm_affine, feature_dropout=feature_dropout, 
                                         rel_pos_bias=rel_pos_bias, max_positions=max_positions,
                                         with_move=time_with_move)

        self.feat_layer = get_mega_block(emb_dim=f_emb_dim, zdim=zdim, hdim=hdim, ndim=ndim,
                                         encoder_ffn_embed_dim=encoder_ffn_embed_dim,
                                         dropout=dropout, attention_dropout=attention_dropout,
                                         hidden_dropout=hidden_dropout, activation=activation,
                                         attention_activation=attention_activation, 
                                         bidirectional=bidirectional, chunk_size=chunk_size,
                                         truncation=truncation, norm_type=norm_type, prenorm=prenorm, 
                                         norm_affine=norm_affine, feature_dropout=feature_dropout, 
                                         rel_pos_bias=rel_pos_bias, max_positions=max_positions,
                                         with_move=feat_with_move)

        
        # self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        # self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(2, 0, 1), None).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feat_layer(y.permute(2, 0, 1), None).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        y = y + cond_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        # import pdb; pdb.set_trace()
        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip


class diff_CSDI_D3M(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.config = config
        self.channels = config['diffusion']["channels"]

        self.diffusion_embedding = DiffusionGFEmbedding(
            embedding_dim=config['diffusion']["diffusion_embedding_dim"],
        )

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)

        self.output_projection3 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection4 = Conv1d_with_init(self.channels, 1, 1)

        nn.init.zeros_(self.output_projection2.weight)
        nn.init.zeros_(self.output_projection4.weight)

        # self.residual_layers = nn.ModuleList(
        #     [
        #         ResidualBlock(
        #             side_dim=config["side_dim"],
        #             channels=self.channels,
        #             diffusion_embedding_dim=config["diffusion_embedding_dim"],
        #             nheads=config["nheads"],
        #         )
        #         for _ in range(int((config["layers"]*1.5)))
        #     ]
        # )
        self.residual_layers = nn.ModuleList(
            [
                EMA_ResidualBlock(
                    side_dim=config['diffusion']["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config['diffusion']["diffusion_embedding_dim"],
                    t_emb_dim=self.channels,
                    f_emb_dim=self.channels,
                    zdim=config['model']['zdim'],
                    hdim=config['model']['hdim'],
                    ndim=config['model']['ndim'],
                    encoder_ffn_embed_dim=config['model']['ffn_embed_dim'],
                    dropout=config['model']['dropout'],
                    attention_dropout=config['model']['att_dropout'],
                    # hidden_dropout=config['hid_dropout'],
                    activation=config['model']['activation'],
                    attention_activation=config['model']['att_activation'],
                    bidirectional=True,
                    chunk_size=config['model']['chunk_size'],
                    truncation=None,
                    norm_type=config['model']['norm_type'],
                    prenorm=bool(config['model']['prenorm']),
                    norm_affine=bool(config['model']['norm_affine']),
                    feature_dropout=bool(config['model']['feat_dropout']),
                    rel_pos_bias=config['model']['rel_pos_bias'],
                    max_positions=config['model']['max_positions'],
                    time_with_move=config['model']['time_move'],
                    feat_with_move=config['model']['feat_move'],
                )
                for _ in range(int((config['diffusion']["layers"]*1.5)))
            ]
        )

        self.deal_skip1 = nn.Conv2d(self.channels, self.channels, 1, 1)
        self.deal_skip2 = nn.Conv2d(self.channels, self.channels, 1, 1)

    # TODO: check output
    def forward(self, x, diffusion_step, cond_info):
        B, inputdim, K, L = x.shape

        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip_1 = []
        skip_2 = []
        for layer in self.residual_layers[:self.config['diffusion']['layers'] // 2]:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip_1.append(self.deal_skip1(skip_connection))
            skip_2.append(self.deal_skip2(skip_connection))

        x1 = x2 = x
        skip1 = []
        skip2 = []
        for i in range(self.config['diffusion']['layers'] // 2):
            layer1 = self.residual_layers[self.config['diffusion']['layers'] // 2 + 2*i]
            layer2 = self.residual_layers[self.config['diffusion']['layers'] // 2 + 2*i + 1]
            x1, skip_connection1 = layer1(x1, cond_info, diffusion_emb)
            x2, skip_connection2 = layer2(x2, cond_info, diffusion_emb)
            skip1.append(skip_connection1)
            skip2.append(skip_connection2)
        


        x1 = torch.sum(torch.stack(skip1 + skip_1), dim=0) / math.sqrt(self.config['diffusion']["layers"])
        x1 = x1.reshape(B, self.channels, K * L)
        x1 = self.output_projection1(x1)  # (B,channel,K*L)
        x1 = F.relu(x1)
        x1 = self.output_projection2(x1)  # (B,1,K*L)
        x1 = x1.reshape(B, K, L)

        x2 = torch.sum(torch.stack(skip2 + skip_2), dim=0) / math.sqrt(self.config['diffusion']["layers"])
        x2 = x2.reshape(B, self.channels, K * L)
        x2 = self.output_projection3(x2)  # (B,channel,K*L)
        x2 = F.relu(x2)
        x2 = self.output_projection4(x2)  # (B,1,K*L)
        x2 = x2.reshape(B, K, L)

        return (x1,), x2

if __name__ == "__main__":
    md = DiffusionEmbedding(num_steps=50, embedding_dim=128)
    diffusion_step = torch.randint(0, 50, (32,)).long()
    res = md(diffusion_step)
    print(res.shape)
    md1 = GaussianFourierProjection(embedding_size=128)
    diff_t = torch.rand((32,)).float()
    res1 = md1(diff_t)
    print(res1.shape)