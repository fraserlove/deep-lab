"""
Basic (byte-level) Byte Pair Encoding (BPE) Tokenizer.

Based off the GPT-2 tokeniser:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

Does not handle any special tokens or the regular expression splitting pattern.
"""

from tokeniser.base import Tokeniser, consecutive_pairs, replace_pair

class BasicTokeniser(Tokeniser):
    """Basic BPE Tokeniser."""

    def __init__(self, tokeniser_file: str = None):
        super().__init__()
        if tokeniser_file is not None:
            self.load(tokeniser_file)

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Train on a text and build a vocabulary of size vocab_size."""
        assert vocab_size >= 256
        n_merges = vocab_size - 256
        tokens = list(text.encode('utf-8'))
        merges = {} # Dictionary to store the merges
        vocab = {i: bytes([i]) for i in range(256)}

        # Merge the most frequent pair n_merges times to create new tokens
        for i in range(n_merges):
            # Find the most frequent consecutive pair of tokens
            freq_pairs = consecutive_pairs(tokens)
            max_pair = max(freq_pairs, key=freq_pairs.get)
            # Create a new token and assign it to an unused integer
            new_token = 256 + i
            tokens = replace_pair(tokens, max_pair, new_token)
            # Store the merge and the new token in the vocab
            merges[max_pair] = new_token
            vocab[new_token] = vocab[max_pair[0]] + vocab[max_pair[1]]
            if verbose:
                print(f'{i+1}/{n_merges}: {max_pair} -> {new_token}')

        self.merges = merges
        self.vocab = vocab

    def encode(self, text: str) -> list[int]:
        """Encode a string into a sequence of tokens."""
        tokens = list(text.encode('utf-8'))
        while len(tokens) > 1:
            freq_pairs = consecutive_pairs(tokens)
            # Find the most frequent consecutive pair that has been merged
            most_freq = min(freq_pairs, key=lambda pair: self.merges.get(pair, float('inf')))
            if most_freq not in self.merges:
                break # No more merges to apply
            # Merge the pair into a new token
            new_token = self.merges[most_freq]
            tokens = replace_pair(tokens, most_freq, new_token)
        return tokens
    
    def decode(self, tokens: list[int]) -> str:
        """Decode a sequence of tokens into a string."""
        bytes_ = b''.join(self.vocab[token] for token in tokens)
        text = bytes_.decode('utf-8', errors='replace') # Replace unknown characters
        return text