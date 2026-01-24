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
        end: int,
        special_tokens: Optional[List[str]]
) -> Counter:
    word_freq = Counter()

    splitter = re.compile(PAT)

    special_token_split_pattern = None
    if special_tokens:
        pattern_str = "|".join(re.escape(t) for t in special_tokens)
        special_token_split_pattern = re.compile(f"({pattern_str})")
        special_tokens_set = set(special_tokens)

    with open(input_path, 'rb') as f:
        f.seek(start)
        size_to_read = end - start

        if size_to_read <= 0:
            return word_freq
        
        chunk = f.read(size_to_read)
        text = chunk.decode('utf-8', errors="ignore")

        if special_token_split_pattern:
            segments = special_token_split_pattern.split(text)
        else:
            segments = [text]
        
        for segment in segments:
            if not segment:
                continue

            if segment in special_tokens_set:
                    # (don't split into bytes)
                    # e.g., (b'<|endoftext|>',)
                    word_freq[ (segment.encode('utf-8'),) ] += 1
                    continue
            
            words = splitter.findall(segment)
            word_freq.update(
                tuple(bytes([b]) for b in word.encode("utf-8"))
                for word in words
            )
    return word_freq

def get_stats(
        vocab: Dict[Tuple[bytes, ...], int]
) -> Dict[Tuple[bytes, bytes], int]:
    pairs = Counter()
    for word_tuple, freq in vocab.items():
        for i in range(len(word_tuple) - 1):
            pair = (word_tuple[i], word_tuple[i+1])
            pairs[pair] += freq
    return pairs

def merge_word(
        word: Tuple[bytes, ...],
        pair: Tuple[bytes, bytes]
) -> Tuple[bytes, ...]:
    new_word = []
    p0, p1 = pair
    merged_byte = p0 + p1
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == p0 and word[i+1] == p1:
            new_word.append(merged_byte)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)
    

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
        process_chunk_args.append((input_path, start, end, special_tokens))
    
    word_freqs = Counter()

    with multiprocessing.Pool(processes=num_processes) as pool:
        chunk_freqs = pool.starmap(process_chunk, process_chunk_args)

        for freq in chunk_freqs:
            word_freqs.update(freq)

    end_time = time.time()

    # print(f"pretokenization time: {end_time - start_time:.4f} sec")
    # print(f"freq of ' the' : {word_freqs[(b' ', b't', b'h', b'e')]}")

    # bpe
    vocab = word_freqs.copy()
    merges = []
    pairs_freq = get_stats(vocab)

    final_vocab_list = [bytes([i]) for i in range(256)]
    for st in special_tokens:
        final_vocab_list.append(st.encode("utf-8"))

    num_merges = vocab_size - len(final_vocab_list)

    if num_merges < 0:
            num_merges = 0

    for i in range(num_merges):
        if not pairs_freq:
            break
        best_pair = max(pairs_freq, key=lambda x: (pairs_freq[x], x))
        if pairs_freq[best_pair] < 1:
            break

        merges.append(best_pair)
        p0, p1 = best_pair
        final_vocab_list.append(p0 + p1)
        new_vocab = {}

        for word, freq in vocab.items():
            if p0 not in word:
                new_vocab[word] = freq
                continue
            new_word_tuple = merge_word(word, best_pair)

            if new_word_tuple != word:
                for j in range(len(word) - 1):
                    pair = (word[j], word[j+1])
                    pairs_freq[pair] -= freq
                    if pairs_freq[pair] == 0:
                        del pairs_freq[pair]
                for j in range(len(new_word_tuple) - 1):
                    pair = (new_word_tuple[j], new_word_tuple[j+1])
                    pairs_freq[pair] += freq
                
                new_vocab[new_word_tuple] = freq
            else:
                new_vocab[word] = freq

        vocab = new_vocab

    final_vocab_map = {idx: token for idx, token in enumerate(final_vocab_list)}
    return final_vocab_map, merges


def save_to_disk(
        vocab: Dict[int, bytes], 
        merges: List[Tuple[bytes, bytes]], 
        vocab_path="vocab.json", 
        merges_path="merges.txt"
):
    vocab_str_map = {
        k: v.decode('utf-8', errors='replace') for k, v in vocab.items()
    }
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_str_map, f, indent=2, ensure_ascii=False)
        
    with open(merges_path, "w", encoding="utf-8") as f:
        for p1, p2 in merges:
            f.write(f"{p1.decode('utf-8', errors='replace')} {p2.decode('utf-8', errors='replace')}\n")


def main():
    test_file = "./data/TinyStoriesV2-GPT4-valid.txt"
    # test_file = "./data/TinyStoriesV2-GPT4-train.txt"

    start_time = time.time()
    vocab, merges = train_bpe(
        input_path=test_file,
        vocab_size=1000,
        special_tokens=["<|endoftext|>"]
    )
    end_time = time.time()
    elapsed = end_time - start_time

    print(f"vocab size: {len(vocab)}")
    print(f"merges size: {len(merges)}")
    # print("Top 5 Merges:", merges[:5])

    print(f"total training time: {elapsed:.4f} seconds ({elapsed/60:.4f} mins)")

    longest_token_id = max(vocab, key=lambda k: len(vocab[k]))
    longest_token_bytes = vocab[longest_token_id]
    print(f"longest token {longest_token_id}: {longest_token_bytes}")
    print(f"longest token length: {len(longest_token_bytes)}")

    save_to_disk(vocab, merges)
    

if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()
    
    main()
    
    pr.disable()
    
    s = io.StringIO()
    # Sort by cumulative time to see which functions took the most time
    # (including time spent in sub-functions)
    sortby = pstats.SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(5) # Print top 5 bottlenecks
    print(s.getvalue())