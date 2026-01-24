import os
from typing import List, Dict, Tuple, BinaryIO, Optional
import multiprocessing
import time
from collections import Counter
import regex as re
import cProfile
import pstats
import io
import json

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(
            self, 
            vocab: Dict[int, bytes], 
            merges: List[Tuple[bytes, bytes]], 
            special_tokens: Optional[List[str]]=None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []

        self.token_to_id = {v: k for k, v in vocab.items()}
        self.merges_to_id = {pair: i for i, pair in enumerate(merges)}

        if self.special_tokens:
            current_max_id = max(self.vocab.keys()) if self.vocab else -1
            next_id = current_max_id + 1
            for token in special_tokens:
                token_bytes = token.encode("utf-8")
                if token_bytes not in self.token_to_id:
                    self.vocab[next_id] = token_bytes
                    self.token_to_id[token_bytes] = next_id
                    next_id += 1

        self.splitter = re.compile(PAT)

        self.special_token_split_pattern = None
        if self.special_tokens:
            pattern_str = "|".join(re.escape(t) for t in self.special_tokens)
            self.special_token_split_pattern = re.compile(f"({pattern_str})")
            self.special_tokens_set = set(special_tokens)

    @classmethod
    def from_files(
            cls,
            vocab_filepath: str,
            merges_filepath: str,
            special_tokens: Optional[List[str]] = None
    ):
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_raw = json.load(f)
        
        vocab = {int(k): v.encode("utf-8") for k, v in vocab_raw.items()}

        merges = []

        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue

                parts = line.split(" ")
                if len(parts) == 2:
                    p1 = parts[0].encode("utf-8")
                    p2 = parts[1].encode("utf-8")
                    merges.append((p1, p2))
        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> List[int]:
        ids = []
        return ids