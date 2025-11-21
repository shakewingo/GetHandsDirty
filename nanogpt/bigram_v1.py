# Based on Karpathy's video "Let's build GPT"

import torch
import torch.nn as nn
from torch.nn import functional as F


batch_size = 64  # num of seq be processed parallelly
block_size = 256  # 8 chars maximumally per batch, or max seq length
max_iters = 5000
learning_rate = 3e-4
eval_iters = 200
eval_interval = 500
n_embd = 384
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_head = 6
n_layers = 4
dropout = 0.2

torch.manual_seed(1234)

# 1. Intro a character-level small GPT

# # We always start with a dataset to train on. Let's download the tiny shakespeare dataset
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# read it in to inspect it
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("length of dataset in characters: ", len(text))

# let's look at the first 1000 characters
# print(text[:1000])

chars = list(sorted(set(text)))
vocab_size = len(chars)
#print(''.join(chars))
print(vocab_size)

# Create a simple tokenizer, i.e. character-level
# Other tokenizer e.g. SequencePiece -> sub-word unit level, tiktoken -> large total vocab; basically total vocab size and seq of integer is a trade-off
abcd = { ch:i for i,ch in enumerate(chars) }
dcba = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [abcd[i] for i in s]
decode = lambda l: ''.join([dcba[i] for i in l]) # decoder: take a list of integers, output a string
# print(encode('hello world'))
# print(decode([46, 43, 50, 50, 53, 1, 61, 53, 56, 50, 42]))

data = torch.tensor(encode(text), dtype=torch.long)
#print(data.shape, data.dtype)
# print(data[:1000]) # the 1000 characters we looked at earier will to the GPT look like this

train_len = int(0.9 * len(data))
train_data = data[:train_len]
val_data = data[train_len:]

# # Create context and target, specifically for batch, context (token_0_integer, token_1_integer, token_2_integer, ..), target (token_1_integer, token_2_integer, ...), i.e. just to predict next token
# x= train_data[:block_size]
# y = train_data[1: block_size+1]
# for t in range(block_size):
#     context = x[:t+1]
#     target = y[t]
#     print(f"when input is {context} the target: {target}")


@torch.no_grad()   # meaning no need to compute gradients for the seek of speed
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):   # average loss over eval_iters batches
            X, Y = get_batch_data(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_batch_data(split='train'):
  data = train_data if split=='train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i: i+block_size] for i in ix])
  y = torch.stack([data[i+1: i+block_size+1] for i in ix])
  x, y = x.to(device), y.to(device) # move to GPU if available
  return x, y

# xb, yb = get_batch_data('train')
# print('inputs:')
# print(xb.shape)
# print(xb)
# print('targets:')
# print(yb.shape)
# print(yb)

# print('----')
# for b in range(batch_size): # batch dimension
#   for t in range(block_size):
#       context = xb[b, :t+1]
#       target = yb[b, t]
#       print(f"when input is {context} the target: {target}")

class Head(nn.Module):
     # Expand Version 2 for self-attention
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, C) -> (B, T, head_size)
        q = self.query(x)
        v = self.value(x)
        wei = k @ q.transpose(-2, -1) * C **-0.5  # Purpose: if torch.tensor([] * 100) -> softmax -> will make larger value even larger, so rescale it
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v # (B, T, T) -> (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)  # TODO: why this needs to be projected?
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

# class LayerNorm1d: 
#   # Used to be BatchNorm1d, but instead of normalize column, here normalize row, so no more normalize acorss training exmaples are needed
#   def __init__(self, dim, eps=1e-5, momentum=0.1):
#     self.eps = eps
#     self.gamma = torch.ones(dim)
#     self.beta = torch.zeros(dim)

#   def __call__(self, x):
#     # calculate the forward pass
#     xmean = x.mean(1, keepdim=True) # layer mean
#     xvar = x.var(1, keepdim=True) # layer variance
#     xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
#     self.out = self.gamma * xhat + self.beta
#     return self.out

#   def parameters(self):
#     return [self.gamma, self.beta]

class FeedForward(nn.Module):
    # Just a MLP
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # based on transformer's paper
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.LayerNorm(n_embd),  # typically need to add as well
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, num_heads=4):
        super().__init__()
        self.sa = MultiHeadAttention(num_heads, head_size=n_embd//num_heads)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # ResNet part, slightly changed order of LayerNorm and Attention from transformer paper
        x = x + self.ffwd(self.ln2(x))
        return x
        

# Build a simple bigram model (simply predict what next token is, tokens won't talk to each other yet!)
class BigramLanguageModel(nn.Module):
      def __init__(self, n_embd):
        super().__init__()
        # each token directly reads off the logits for the next token from the learnable lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding_table = nn.Embedding(block_size, n_embd)
        # self.ma_head = MultiHeadAttention(num_heads=4, head_size=n_embd//4)
        # self.ffwd = FeedForward(n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd) for _ in range(n_layers)])  # so it can compute may times
        self.lm_head = nn.Linear(n_embd, vocab_size)

      def forward(self, idx, targets=None):
        B, T = idx.shape
        logits = self.token_embedding_table(idx) # input shape: (B, T), output shape: (B, T, C)
        pos_emb = self.pos_embedding_table(torch.arange(T, device=device)) # input shape: (T), output shape: (T, C)
        x = logits + pos_emb
        # x = self.ma_head(x)
        # x = self.ffwd(x)
        logits = self.lm_head(x) # input shape: (B, T, C), output shape: (B, T, vocab_size)
        if targets is None:
          loss = None
        else:
          B, T, C = logits.shape
          logits = logits.view(B*T, C)
          targets = targets.view(B*T)
          loss = F.cross_entropy(logits, targets)
        return logits, loss

      # inference
      def generate_next_token(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
          logits, loss = self.forward(idx[:,-block_size:])   # idx[:, -block_size:] here is to prevent embedding table run out of scope
          logits = logits[:, -1, :] # focus on the last time stamp
          probs = F.softmax(logits, dim=-1) # output: (B, C)
          next_idx = torch.multinomial(probs, num_samples=1) # output: (B, 1)
          idx = torch.cat((idx, next_idx), dim=1) # (B, T+1)
        return idx

# Train
model = BigramLanguageModel(n_embd)
model = model.to(device) # move to GPU if available
print(f'Training on device: {device}')
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for iter in range(max_iters): 
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch_data('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

# Start getting shakespare-like result
# Create into GPU if available
generated_idx = model.generate_next_token(idx=torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=500)
print(generated_idx)
result = decode(generated_idx[0].tolist())
print(result)



