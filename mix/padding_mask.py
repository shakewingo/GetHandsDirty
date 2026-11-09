import math
import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model=128, num_heads=4):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, attention_mask):
        # x:              (B, L, D)
        # attention_mask: (B, L), 1=valid token, 0=padding
        B, L, D = x.shape
        valid = attention_mask.bool()

        # 1. Compute Q, K, and V, then split into multiple heads
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # Each has shape (B, L, H, d)

        q = q.transpose(1, 2)        # (B, H, L, d)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 2. Attention logits: rows are Queries, columns are Keys
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # scores: (B, H, L, L)

        # 3. Causal mask: allow attention only to current and earlier positions
        causal_allowed = torch.ones(
            L, L, dtype=torch.bool, device=x.device
        ).tril()
        # shape: (L, L)

        # 4. Padding mask: prevent all Queries from attending to padding Keys
        key_valid = valid[:, None, None, :]
        # shape: (B, 1, 1, L), broadcast across heads and Query positions

        allowed = causal_allowed[None, None, :, :] & key_valid
        # shape: (B, 1, L, L)

        scores = scores.masked_fill(~allowed, float("-inf"))

        # Prevent NaNs from Softmax when an entire row is -inf
        # This can occur for all-padding samples or leading left-padding positions
        row_has_key = allowed.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(~row_has_key, 0.0)

        # 5. Normalize over Keys; compute Softmax in FP32 for numerical stability
        weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        weights = weights.masked_fill(~allowed, 0.0)

        # 6. Compute the weighted sum of Values
        out = weights @ v              # (B, H, L, d)
        out = out.transpose(1, 2).contiguous().reshape(B, L, D)
        out = self.out_proj(out)

        # Zero out outputs at padding Query positions
        out = out.masked_fill(~valid[:, :, None], 0.0)

        return out, weights


# Example: the second sample ends with two padding tokens
torch.manual_seed(0)

token_ids = torch.tensor([
    [1, 2, 3, 4],
    [5, 6, 0, 0],
])
attention_mask = token_ids.ne(0)

embedding = nn.Embedding(10, 128, padding_idx=0)
x = embedding(token_ids)

attention = CausalSelfAttention(d_model=128, num_heads=4)
out, weights = attention(x, attention_mask)

print(out.shape)      # torch.Size([2, 4, 128])
print(weights.shape)  # torch.Size([2, 4, 4, 4])

# Attention weights for the first head of the second sample
print(weights[1, 0].detach().round(decimals=3))

# Takesaways, attention_mask is usually a operation that returns True/False，after that, via scores.masked_fill(~allowed, float("-inf")) to process non-allowed positions.
# the padding mask first masked key column，then the row of padding q even computes weights, result will become zero。
# Additionally，attention mask is not equal to loss mask：during training, still need to make padding not being considered in loss，e.g. CrossEntropyLoss(ignore_index=0)。