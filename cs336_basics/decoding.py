import torch
import argparse
import os

from .transformer_lm import TransformerLM
from .tokenizer import Tokenizer
from .checkpointing import load_checkpoint
from .softmax import softmax

def top_p_sampling(
        logits: torch.Tensor,
        top_p: float=0.9,
        temperature: float=1.0
) -> torch.Tensor:
    logits = logits / temperature

    probs = softmax(logits, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

        cum_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cum_probs > top_p

        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :1].clone()

        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)

        probs = probs.masked_fill(indices_to_remove, 0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True)

    next_token = torch.multinomial(probs, num_samples=1)
    return next_token

@torch.no_grad()
def generate(
    model: TransformerLM,
    prompt_tokens: torch.Tensor,
    max_new_tokens: int,
    max_seq_len: int,
    temperature: float=1.0,
    top_p: float=0.9,
    eos_id: int=None
):
    model.eval()
    curr_seq = prompt_tokens
    for _ in range(max_new_tokens):
        idx_cond = curr_seq if curr_seq.size(1) <= max_seq_len else curr_seq[:, -max_seq_len:]
        logits = model(idx_cond)

        logits = logits[:, -1, :]

        next_token = top_p_sampling(logits, top_p, temperature)

        curr_seq = torch.cat((curr_seq, next_token), dim=-1)

        if eos_id is not None and next_token.item() == eos_id:
            break

    return curr_seq

def main():
    parser = argparse.ArgumentParser()

    # model checkoint
    parser.add_argument("--checkpoint", type=str, required=True, help="path to model checkpoint")
    
    # tokenizer
    parser.add_argument("--vocab_path", type=str, required=True, help="path to tokenizer vocab file")
    parser.add_argument("--merges_path", type=str, required=True, help="path to tokenizer merges file")
    
    # generation config
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="input text prompt")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="sampling temperature (lower is more deterministic)")
    parser.add_argument("--top_p", type=float, default=0.9, help="nucleus sampling probability threshold")
    
    # Model Architecture (Must match training!)
    parser.add_argument("--d_model", type=int, default=512, help="embedding dimension (d_model)")
    parser.add_argument("--n_layers", type=int, default=8, help="number of transformer layers (num_layers)")
    parser.add_argument("--n_heads", type=int, default=8, help="number of attention heads (num_heads)")
    parser.add_argument("--d_ff", type=int, default=None, help="feed forward dimension. Defaults to 4 * dim if not provided")
    parser.add_argument("--vocab_size", type=int, default=32000, help="vocabulary size")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="maximum sequence length (context_length)")
    parser.add_argument("--device", type=str, default="mps" if torch.mps.is_available() else "cpu")

    args = parser.parse_args()
    device = torch.device(args.device)

    tokenizer = Tokenizer.from_files(args.vocab_path, args.merges_path, special_tokens=["<|endoftext|>"])

    d_ff = args.d_ff if args.d_ff is not None else 4 * args.d_model

    model_config = {
        "vocab_size": args.vocab_size,
        "context_length": args.max_seq_len,
        "num_layers": args.n_layers,
        "d_model": args.d_model,
        "num_heads": args.n_heads,
        "d_ff": d_ff,
        "device": device
    }

    print(f"tarnsformerLM config: {model_config}")
    model = TransformerLM(**model_config)
    model.to(device)

    print(f"loading from checkpoint {args.checkpoint}...")
    dummy_optimizer = torch.optim.AdamW(model.parameters()) 
    load_checkpoint(args.checkpoint, model, dummy_optimizer)

    prompt_ids = tokenizer.encode(args.prompt)
    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

    print("generating...")

    eos_id = tokenizer.token_to_id["<|endoftext|>".encode("utf-8")]

    generated_ids = generate(
        model=model, 
        prompt_tokens=prompt_tensor, 
        max_new_tokens=args.max_new_tokens, 
        max_seq_len=args.max_seq_len,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_id=eos_id
    )

    output_ids = generated_ids.squeeze(0).tolist()
    decoded_text = tokenizer.decode(output_ids)

    print(decoded_text)

if __name__ == "__main__":
    main()

    