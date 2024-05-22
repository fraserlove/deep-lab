import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu' # Use GPU if available

# B - batch size, T - block size (time step), C - embedding dimension, C' - vocab size, H - head size

# Hyperparameters
batch_size = 32 # Sequences to process in parallel
block_size = 128 # Maximum context length for predictions
max_iters = 5000 # Iterations to train the model
eval_iters = 200 # Iterations to average the loss over
n_embed = 32 # Embedding dimensions
n_head = 4 # Heads in the multi-head self-attention
n_block = 4 # Number of transformer blocks
dropout = 0.2 # Dropout probability
lr = 5e-3 # Learning rate

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Unique characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Mapping from characters to integers and vice versa
char_to_int = {c: i for i, c in enumerate(chars)}
int_to_char = {i: c for i, c in enumerate(chars)}

encode = lambda x: [char_to_int[c] for c in x] # x: str -> list[int]
decode = lambda x: ''.join([int_to_char[i] for i in x]) # x: list[int] -> str

# Train and validation splits
data = torch.tensor(encode(text), dtype=torch.long)
split = int(len(data) * 0.9) # 90% train, 10% val
train_data, val_data = data[:split], data[split:]

def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a random batch of context and target sequences."""
    data = train_data if split == 'train' else val_data
    # Randomly sample batch_size number of starting indices
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model: nn.Module) -> dict[str, float]:
    """Estimate the mean loss of the model on the train and val sets."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out    

class Head(nn.Module):
    """Single head of self-attention."""
    
    def __init__(self, head_size: int):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False) # (B,T,C) -> (B,T,H)
        self.query = nn.Linear(n_embed, head_size, bias=False) # (B,T,C) -> (B,T,H)
        self.value = nn.Linear(n_embed, head_size, bias=False) # (B,T,C) -> (B,T,H)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.Dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # Compute attention scores ('affinities')
        W = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5) # (B,T,H) @ (B,H,T) -> (B,T,T)
        W = W.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        W = F.softmax(W, dim=-1)
        W = self.Dropout(W)
        # Perform the attention-weighted sum
        v = self.value(x)
        out = W @ v # (B,T,T) @ (B,T,H) -> (B,T,H)
        return out

class MultiHead(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, n_head: int, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(head_size * n_head, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):
    """Single non-linear feed-forward layer."""

    def __init__(self, n_embed: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
class Block(nn.Module):
    """Transformer block. A multi-head self-attention layer and a feed-forward layer."""
    
    def __init__(self, n_embed: int, n_head: int):
        super().__init__()
        self.sa_heads = MultiHead(n_head, n_embed // n_head)
        self.feed_forward = FeedForward(n_embed)
        self.layer_norm1 = nn.LayerNorm(n_embed)
        self.layer_norm2 = nn.LayerNorm(n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply self-attention and feed-forward layers with residual connections and layer normalisation.
        x = self.layer_norm1(self.sa_heads(x) + x)
        x = self.layer_norm2(self.feed_forward(x) + x)
        return x

class GPTLanguageModel(nn.Module):
    """GPT Decoder model. Consists of an embedding layer, transformer blocks, and a linear head."""

    def __init__(self):
        super().__init__()
        self.token_embed_table = nn.Embedding(vocab_size, n_embed) # (B,T) -> (B,T,C)
        self.position_embed_table = nn.Embedding(block_size, n_embed) # (T) -> (T,C)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_block)])
        self.layer_norm = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size) # (B,T,C) -> (B,T,C')

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = x.shape
        token_embed = self.token_embed_table(x)
        position_embed = self.position_embed_table(torch.arange(T, device=device))
        embed = token_embed + position_embed
        embed = self.blocks(embed)
        embed = self.layer_norm(embed)
        logits = self.lm_head(embed)

        if y is None:
            loss = None
        else:
            B, T, C = logits.shape
            # Flatten batch and sequence dimensions to use F.cross_entropy
            logits = logits.view(B*T, C)
            y = y.view(B*T)
            loss = F.cross_entropy(logits, y)
        return logits, loss

    def generate(self, x: torch.Tensor, max_tokens: int) -> torch.Tensor:
        for _ in range(max_tokens):
            # Crop to the last block_size tokens
            x_last = x[:, -block_size:]
            # Get the previous predictions
            logits, _ = self(x_last)
            # Keep only the last prediction
            logits = logits[:, -1, :] # (B,C)
            # Apply softmax to convert logits into probabilities
            probs = F.softmax(logits, dim=-1) # (B,C)
            # Sample from the probability distribution
            x_next = torch.multinomial(probs, num_samples=1) # (B,1)
            # Concatenate the new prediction to the previous context
            x = torch.cat([x, x_next], dim=1) # (B,T+1)
        return x

model = GPTLanguageModel()
model = model.to(device)

total_params = sum(param.numel() for param in model.parameters())
print(f'Model parameters: {total_params}')

# Training the model
optimiser = torch.optim.AdamW(model.parameters(), lr=lr)

for i in range(max_iters):
    
    if i % (max_iters // 10) == 0 or i == max_iters - 1:
        losses = estimate_loss(model)
        print(f'iteration {i}, train loss: {losses["train"]:.4f}, val loss: {losses["val"]:.4f}')

    # Get a batch of context and target sequences
    xb, yb = get_batch('train')

    # Compute the gradients and update the weights
    _, loss = model(xb, yb) # Forward pass
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    optimiser.step()

# Generate
context = torch.zeros((1, 1), dtype=torch.long, device=device)
with open('output.txt', 'w') as f:
    f.write(decode(model.generate(context, max_tokens=512)[0].tolist()))