from gpt.tokeniser.gpt import GPTTokeniser

vocab_size = 10000 # Target number of unique tokens

# Special tokens to be added to the vocabulary. GPT-4 uses these tokens
special_tokens = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}

# Load new text from a file
with open('openwebtext-10k.txt', 'r', encoding='utf-8') as file:
    text = file.read()

tokeniser = GPTTokeniser()
tokeniser.train(text, vocab_size=vocab_size, verbose=True)

# Register special tokens
tokeniser.register_special_tokens(special_tokens)

tokeniser.save('openwebtext-10k')