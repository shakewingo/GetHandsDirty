import math

import torch
import torch.nn as nn


def get_config():
    return {'batch_size': 8, 'num_epochs': 20, 'lr': 0.0001, 'seq_len': 350, 'd_model': 512, }


class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, d_model: int, dropout: float):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(0, seq_len).float().unsqueeze(1)  # (seq_len, 1)
        pe = torch.zeros(seq_len, d_model)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)  # (seq_len, d_model)
        pe[:, 1::2] = torch.cos(position * div)  # (seq_len, d_model)

        # Note that: unsqueeze(0): add one more dim in row, unsqueeze(1): add one more dim in col
        pe = pe.unsqueeze(0)  # (seq_len, d_model) -> (1, seq_len, d_model)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x += self.pe.requires_grad_(False)  # TODO: need to convert to (batch, seq_len, d_model)?
        out = self.dropout(x)
        return out


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.l1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.l2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        out = self.l2(self.dropout(torch.relu(self.l1(x))))
        return out


class LayerNormalization(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(d_model))  # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(d_model))  # bias is a learnable parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        std = x.std(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        return self.alpha * (x - mean / math.sqrt(std + 10 ** -6)) + self.bias


# layer_norm = nn.LayerNormalization(d_model)
# layer_norm(embedding=(batch, seq_len, d_model))


class ResidualConnection(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(d_model)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)
        assert d_model % h == 0, 'd_model is not divisible by h'
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.d_k = self.d_model // h

    @staticmethod
    def attention(q_i, k_i, v_i, mask, dropout: nn.Dropout):
        # q_i: (batch, h, seq_len, d_k), k_i: (batch, h, seq_len, d_k)
        d_k = q_i.shape[-1]
        attention_score = q_i @ k_i.transpose(-2, -1) / math.sqrt(d_k)  # (batch, h, seq_len, seq_len)
        if mask:
            attention_score.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_score.softmax(dim=-1)  # (batch, h, seq_len, seq_len)
        if dropout:
            attention_scores = dropout(attention_scores)
        # return attention_scores for visluazation
        return (attention_scores @ v_i), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # (batch, seq_len, d_model) -> (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        # concentrate multi-head: (batch, h, seq_len, d_k) -> (batch, seq_len, h, d_k) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        return self.w_o(x)


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(ResidualConnection(d_model, dropout) for _ in range(2))

    def forward(self, x, src_mask):
        x = self.residual_connections[0](self.self_attention_block(x, x, x, src_mask))  # TODO: different than org
        x = self.residual_connections[1](self.feed_forward_block(x))
        return x


class Encoder(nn.Module):
    def __init__(self, d_model: int, blocks: nn.ModuleList) -> None:
        super().__init__()
        self.blocks = blocks
        self.norm = LayerNormalization(d_model)

    def forward(self, x, src_mask):
        for block in self.blocks:
            x = block(x, src_mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, self_attention_block: MultiHeadAttentionBlock,
                 cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock,
                 dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(ResidualConnection(d_model, dropout) for _ in range(3))

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](self.self_attention_block(x, x, x, tgt_mask))  # TODO: different than org
        x = self.residual_connections[1](self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](self.feed_forward_block(x))
        return x


class Decoder(nn.Module):
    def __init__(self, d_model: int, blocks: nn.ModuleList) -> None:
        super().__init__()
        self.blocks = blocks
        self.norm = LayerNormalization(d_model)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for block in self.blocks:
            x = block(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


# TODO: last linear part? what about softmax?
class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings,
                 src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int,
                      d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1,
                      d_ff: int = 2048) -> Transformer:
    src_embed = InputEmbeddings(src_vocab_size, d_model)
    tgt_embed = InputEmbeddings(tgt_vocab_size, d_model)
    src_pos = PositionalEncoding(src_seq_len, d_model, dropout)
    tgt_pos = PositionalEncoding(tgt_seq_len, d_model, dropout)

    # build encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block,
                                     feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # create encoder and encoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(encoder_blocks))

    # create projection layter
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer
