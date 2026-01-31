import os
import time
import numpy as np
from typing import List
import multiprocessing
import functools

from tokenizer import Tokenizer

TS_TRAIN_PATH = "./data/TinyStoriesV2-GPT4-train.txt"
TS_VAL_PATH = "./data/TinyStoriesV2-GPT4-valid.txt"

OUTPUT_DIR = "./dataset"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_tokenizer(
        name: str
) -> Tokenizer:
    vocab_path = os.path.join(OUTPUT_DIR, f"{name}_vocab.json")
    merges_path = os.path.join(OUTPUT_DIR, f"{name}_merges.txt")
    if os.path.exists(vocab_path) and os.path.exists(merges_path):
        return Tokenizer.from_files(vocab_path, merges_path, special_tokens=["<|endoftext|>"])
    
    raise FileNotFoundError(f"vocab_path {name}_vocab.json or merges_path {name}_merges.txt does not exist")

def _mp_encode_chunk(tokenizer: Tokenizer, text: str) -> List[int] :
    return tokenizer.encode(text)

def encode_file_to_bin(
        tokenizer: Tokenizer,
        input_path: str,
        output_path: str
):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} not found")
    
    print(f"processing {input_path} -> {output_path}...")

    assert len(tokenizer.vocab) < 65535

    token_count = 0
    start_time = time.time()

    chunk_size = 5 * 1024 * 1024
    num_processes = os.cpu_count() or 4

    def chunk_reader():
        with open(input_path, 'r', encoding="utf-8", errors="ignore") as f_in:
            while True:
                text_chunk = f_in.read(chunk_size)
                if not text_chunk:
                    break
                yield text_chunk

    with open(output_path, "wb") as f_out:
        with multiprocessing.Pool(processes=num_processes) as pool:

            encode_func = functools.partial(_mp_encode_chunk, tokenizer)

            chunk_size = 1024 * 1024 # 1MB

            for ids in pool.imap(encode_func, chunk_reader(), chunksize=1):
                if not ids:
                    continue

                arr = np.array(ids, dtype=np.uint16)
                f_out.write(arr.tobytes())
                token_count += len(ids)

                if token_count % 1000000 == 0:
                    print(f"processed {token_count/1000000:.1f}M tokens...")

    end_time = time.time()
    print(f"\n\nsaved {token_count} tokens to {output_path}")
    print(f"time taken: {end_time - start_time:.2f}s")
    print(f"file size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")

def experiment_stats(tokenizer: Tokenizer):

    some_text = "The quick brown fox jumps over the lazy dog. " * 5000 
    start = time.time()
    _ = tokenizer.encode(some_text)
    end = time.time()
    bytes_processed = len(some_text.encode("utf-8"))
    throughput = bytes_processed / (end - start)
    print(f"throughput: {throughput:,.2f} bytes/second")

    if os.path.exists(TS_TRAIN_PATH):
        with open(TS_TRAIN_PATH, 'r', encoding='utf-8') as f:
            sample_text = f.read(10000)

    ids = tokenizer.encode(sample_text)
    orig_bytes = len(sample_text.encode("utf-8"))
    ratio = orig_bytes / len(ids) if ids else 0
    print(f"compression Ratio: {ratio:.2f} bytes/token (higher is better)")

def main():
    tokenizer = get_tokenizer("TinyStories")
    experiment_stats(tokenizer)
    train_bin_path = os.path.join(OUTPUT_DIR, "train.bin")
    encode_file_to_bin(tokenizer, TS_TRAIN_PATH, train_bin_path)

    val_bin_path = os.path.join(OUTPUT_DIR, "val.bin")
    encode_file_to_bin(tokenizer, TS_VAL_PATH, val_bin_path)

if __name__ == "__main__":
    main()
