import os
from typing import List, Dict, Tuple, BinaryIO, Optional, Iterable, Iterator
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
            # sort by length
            # special_tokens=["<|endoftext|>", "<|endoftext|><|endoftext|>"])
            # test_string = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            pattern_str = "|".join(re.escape(t) for t in sorted_tokens)
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
    
    def _bpe_merge(self, token_bytes: List[bytes]) -> List[bytes]:
        while len(token_bytes) > 1:
            pairs = [(token_bytes[i], token_bytes[i+1]) for i in range(len(token_bytes)-1)]

            existing_pairs = {p: self.merges_to_id[p] for p in pairs if p in self.merges_to_id}

            if not existing_pairs:
                break

            best_pair = min(existing_pairs, key=existing_pairs.get)

            new_token_bytes = []
            i = 0
            while i < len(token_bytes):
                if i < len(token_bytes) - 1 and (token_bytes[i], token_bytes[i+1]) == best_pair:
                    new_token_bytes.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_token_bytes.append(token_bytes[i])
                    i += 1
            token_bytes = new_token_bytes
            
        return token_bytes

    
    def encode(self, text: str) -> List[int]:
        ids = []

        if self.special_token_split_pattern:
            segments = self.special_token_split_pattern.split(text)
        else:
            segments = [text]
        
        for segment in segments:
            if not segment:
                continue
            if segment in self.special_tokens:
                st_bytes = segment.encode("utf-8")
                if st_bytes in self.token_to_id:
                    ids.append(self.token_to_id[st_bytes])
                continue
            words = self.splitter.findall(segment)

            for word in words:
                token_bytes = [bytes([b]) for b in word.encode("utf-8")]
                merged_bytes = self._bpe_merge(token_bytes)
                for b_token in merged_bytes:
                    if b_token in self.token_to_id:
                        ids.append(self.token_to_id[b_token])
        return ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for iter in iterable:
            encoded_ids = self.encode(iter)
            for token_id in encoded_ids:
                yield token_id

    def decode(self, ids: List[int]) -> str:
        bytes_string = b""
        for token_id in ids:
            if token_id in self.vocab:
                bytes_string += self.vocab[token_id]
        return bytes_string.decode("utf-8", errors="replace")