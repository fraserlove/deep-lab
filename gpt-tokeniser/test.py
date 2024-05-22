import os

from gpt import GPTTokeniser

# Special tokens to be added to the vocabulary. GPT-4 uses these tokens
special_tokens = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}

# Load new text from a file
with open('test.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Create a tokeniser and do 64 merges
tokeniser = GPTTokeniser()
vocab_size = 256 + 64
tokeniser.train(text, vocab_size=vocab_size, verbose=True)

# Register special tokens
tokeniser.register_special_tokens(special_tokens)

# Verify that the encode and decode functions are inverses
assert text == tokeniser.decode(tokeniser.encode(text, 'all'))

# Verify that save/load work as expected
tokeniser.save('tmp')
tokeniser = GPTTokeniser()
tokeniser.load('tmp.tkn')

# Verify that the encode and decode functions are inverses
assert text == tokeniser.decode(tokeniser.encode(text, 'all'))

# Remove the temporary file
os.remove('tmp.tkn')