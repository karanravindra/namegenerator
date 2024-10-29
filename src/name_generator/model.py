from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    embedding_dim: int
    num_layers: int
    max_length: int
    q_heads: int | list[int]
    m: int | list[int]
    kv_heads: int | list[int] | None = None
    tie_weights: bool = True

    def __post_init__(self):
        if isinstance(self.q_heads, int):
            self.q_heads = [self.q_heads] * self.num_layers

        if self.kv_heads is not None:
            if isinstance(self.kv_heads, int):
                self.kv_heads = [self.kv_heads] * self.num_layers
            if len(self.kv_heads) != self.num_layers:
                raise ValueError(
                    f"`kv_heads` must be an integer or a list of length {self.num_layers}"
                )

        if isinstance(self.m, int):
            self.m = [self.m] * self.num_layers

        if len(self.q_heads) != self.num_layers:
            raise ValueError(
                f"`q_heads` must be an integer or a list of length {self.num_layers}"
            )
        if len(self.m) != self.num_layers:
            raise ValueError(
                f"`m` must be an integer or a list of length {self.num_layers}"
            )


class MLP(nn.Module):
    def __init__(self, dim, m):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * m)
        self.linear2 = nn.Linear(dim * m, dim)
        self.drop = nn.Dropout(p=0.1, inplace=False)

    def forward(self, x):
        x = self.linear1(x)
        x = nn.functional.gelu(x)
        x = self.drop(x)
        x = self.linear2(x)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, q_heads, kv_heads=None):
        if kv_heads is None:
            kv_heads = q_heads

        super().__init__()
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.head_dim = dim // q_heads

        self.q = nn.Linear(dim, self.head_dim * q_heads)
        self.k = nn.Linear(dim, self.head_dim * kv_heads)
        self.v = nn.Linear(dim, self.head_dim * kv_heads)

        self.out = nn.Linear(dim, dim)

    def forward(self, x, key_padding_mask=None):
        b, t, e = x.shape
        q = self.q(x).reshape(b, t, self.q_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(x).reshape(b, t, self.kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(x).reshape(b, t, self.kv_heads, self.head_dim).permute(0, 2, 1, 3)

        bias = torch.tril(
            torch.ones(q.shape[2], k.shape[2], dtype=torch.bool, device=x.device),
            diagonal=k.shape[2] - q.shape[2],
        )
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).expand(-1, q.shape[2], -1)

            bias = bias.masked_fill(key_padding_mask, False)
            bias = bias.unsqueeze(1).expand(-1, q.shape[1], -1, -1)

        x = (
            F.scaled_dot_product_attention(q, k, v, bias, enable_gqa=True)
            .permute(0, 2, 1, 3)
            .reshape(b, t, e)
        )

        return self.out(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, i: int):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(config.embedding_dim)
        self.norm2 = nn.LayerNorm(config.embedding_dim)
        self.attn = CausalSelfAttention(
            config.embedding_dim, config.q_heads[i], config.kv_heads[i] # type: ignore
        )
        self.mlp = MLP(config.embedding_dim, config.m[i]) # type: ignore

    def forward(self, x, key_padding_mask=None):
        x = x + self.attn(self.norm1(x), key_padding_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(config, i) for i in range(config.num_layers)]
        )
        self.fc = nn.Linear(config.embedding_dim, config.vocab_size)

        self.pos_emb = nn.Parameter(
            torch.randn(1, config.max_length, config.embedding_dim, requires_grad=True)
        )

        if config.tie_weights:
            self.fc.weight = self.embedding.weight

    def forward(self, x, key_padding_mask=None):
        x = self.embedding(x) + self.pos_emb[:, : x.shape[1]]
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, key_padding_mask)
        x = self.fc(x)
        return x