import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu' # Use GPU if available

# B - batch size, T - block size (time step), C - embedding dimension (vocab size)

# Hyperparameters
batch_size = 32 # Sequences to process in parallel
block_size = 128 # Maximum context length for predictions
max_iters = 2500 # Iterations to train the model
eval_iters = 200 # Iterations to average the loss over
lr = 1e-2 # Learning rate

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

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed_table = nn.Embedding(vocab_size, vocab_size) # (B,T) -> (B,T,C)

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.token_embed_table(x)

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
            # Get the previous predictions
            logits, _ = self(x)
            # Keep only the last prediction
            logits = logits[:, -1, :] # (B,C)
            # Apply softmax to convert logits into probabilities
            probs = F.softmax(logits, dim=-1) # (B,C)
            # Sample from the probability distribution
            x_next = torch.multinomial(probs, num_samples=1) # (B,1)
            # Concatenate the new prediction to the previous context
            x = torch.cat([x, x_next], dim=1) # (B,T+1)
        return x

model = BigramLanguageModel()
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
    f.write(decode(model.generate(context, max_tokens=2048)[0].tolist()))