import os
from typing import List, Dict, Tuple, BinaryIO
import multiprocessing
import time
from collections import Counter
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def process_chunk(
        input_path: str,
        start: int,
        end: int
) -> Counter:
    word_freq = Counter()

    splitter = re.compile(PAT)

    with open(input_path, 'rb') as f:
        f.seek(start)
        size_to_read = end - start

        if size_to_read <= 0:
            return word_freq
        
        chunk = f.read(size_to_read)
        text = chunk.decode('utf-8', errors="ignore")
        words = splitter.findall(text)

        word_freq.update(
            tuple(bytes([b]) for b in word.encode("utf-8"))
            for word in words
        )
    return word_freq

def get_stats(
        vocab: Dict[Tuple[bytes, int], int]
) -> Dict[Tuple[bytes, bytes], int]:
    pairs = Counter()
    for word_tuple, freq in vocab.items():
        for i in range(len(word_tuple)-1):
            pair = (word_tuple[i], word_tuple[i+1])
            pairs[pair] += freq
    return pairs

def merge_vocab(
        pair: Tuple[bytes, bytes],
        vocab: Dict[Tuple[bytes, int], int]
) -> Dict[Tuple[bytes, int], int]:
    new_vocab = {}
    p0, p1 = pair
    merged_byte = p0 + p1
    
    for word_tuple, freq in vocab.items():
        new_word = []
        i = 0
        while i < len(word_tuple):
            if i < len(word_tuple) - 1 and word_tuple[i] == p0 and word_tuple[i+1] == p1:
                new_word.append(merged_byte)
                i += 2
            else:
                new_word.append(word_tuple[i])
                i += 1
        new_vocab[tuple(new_word)] = freq
        
    return new_vocab
    

def train_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens: List[str]
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    
    # pre tokenization
    num_processes = os.cpu_count() or 4

    start_time = time.time()
    with open(input_path, 'rb') as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    process_chunk_args = []

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        process_chunk_args.append((input_path, start, end))
    
    word_freqs = Counter()

    with multiprocessing.Pool(processes=num_processes) as pool:
        chunk_freqs = pool.starmap(process_chunk, process_chunk_args)

        for freq in chunk_freqs:
            word_freqs.update(freq)

    end_time = time.time()

    print(f"pretokenization time: {end_time - start_time:.4f} sec")
    print(f"freq of ' the' : {word_freqs[(b' ', b't', b'h', b'e')]}")

    vocab = word_freqs.copy()
    merges = []

    for i in range(vocab_size):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best_pair = max(pairs, key=lambda x: (pairs[x], x))
        merges.append(best_pair)
        vocab = merge_vocab(best_pair, vocab)

    return vocab, merges


def main():
    test_file = "./data/TinyStoriesV2-GPT4-valid.txt"
    
    vocab, merges = train_bpe(
        input_path=test_file,
        vocab_size=20,
        special_tokens=["<|endoftext|>"]
    )

    print(f"vocab size: {len(vocab)}")
    print(f"merges size: {len(merges)}")
    print("Top 5 Merges:", merges[:5])


if __name__ == "__main__":
    main()