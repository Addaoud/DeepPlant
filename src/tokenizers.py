from transformers import EsmTokenizer
from typing import List, Optional


class KmerEsmTokenizer(EsmTokenizer):
    def __init__(self, vocab_file, **kwargs):
        super().__init__(vocab_file=vocab_file, **kwargs)
        # Convert token list to set for fast lookup
        self.token_set = set(self.all_tokens)

    def tokenize(self, text: str, k: Optional[int] = 5, **kwargs) -> List[str]:
        """Efficient overlapping k-mer tokenization with O(1) lookup."""
        unk = self.unk_token
        token_set = self.token_set  # local binding for speed
        return [
            kmer if kmer in token_set else unk
            for i in range(0, len(text) - k + 1, k)
            if len(kmer := text[i : i + k]) == k
        ]
