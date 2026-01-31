import os
import time
import argparse
import numpy as np
import torch

from .transformer_lm import TransformerLM
from .adamw import AdamW
from .cross_entropy import cross_entropy
from .learning_rate_schedule import get_lr_cosine_schedule
from .gradient_clipping import gradient_clipping
from .data_loading import get_batch
from .checkpointing import save_checkpoint, load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser()

    # data params
    parser.add_argument("--data_dir", type=str, required=True, help="directory containing train, val")
    parser.add_argument("--out_dir", type=str, default="checkpoints", help="directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="path to checkpoint to resume from")

    # model hyperparams
    parser.add_argument("--d_model", type=int, default=512, help="embedding dimension (d_model)")
    parser.add_argument("--n_layers", type=int, default=8, help="number of transformer layers (num_layers)")
    parser.add_argument("--n_heads", type=int, default=8, help="number of attention heads (num_heads)")
    parser.add_argument("--d_ff", type=int, default=None, help="feed forward dimension. Defaults to 4 * dim if not provided")
    parser.add_argument("--vocab_size", type=int, default=32000, help="vocabulary size")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="maximum sequence length (context_length)")

    # training hyperparameters
    parser.add_argument("--batch_size", type=int, default=32, help="batch size per device")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="max learning rate")
    parser.add_argument("--max_iters", type=int, default=10000, help="total number of training iterations")
    parser.add_argument("--warmup_iters", type=int, default=1000, help="number of warmup iterations")
    parser.add_argument("--min_lr", type=float, default=3e-5, help="minimum learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="weight decay for optimizer")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="gradient clipping threshold")

    # logging
    parser.add_argument("--device", type=str, default="mps" if torch.mps.is_available() else "cpu", help="device to use (cuda, mps, cpu)")
    parser.add_argument("--eval_interval", type=int, default=500, help="how often to evaluate on validation set")
    parser.add_argument("--save_interval", type=int, default=1000, help="how often to save checkpoints")
    parser.add_argument("--log_interval", type=int, default=10, help="how often to log metrics to console")

    return parser.parse_args()

@torch.no_grad()
def estimate_loss(model, data_dir, batch_size, max_seq_len, device, eval_iters=200):
    out = {}
    model.eval()

    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, batch_size, max_seq_len, device)
            logits = model(X)
            loss = cross_entropy(logits, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"using device: {device}")

    train_data = np.memmap(os.path.join(args.data_dir, 'train.bin'), dtype=np.uint16, mode='r')

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

    optimizer = AdamW(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    iter_num = 0
    best_val_loss = 1e9
    if args.resume:
        print(f"resuming from {args.resume}...")
        iter_num = load_checkpoint(args.resume, model, optimizer)
    
    print(f"model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.3f}M")

    t0 = time.time()

    X, Y = get_batch(train_data, args.batch_size, args.max_seq_len, device)

    while iter_num < args.max_iters:
        lr = get_lr_cosine_schedule(iter_num, args.learning_rate, args.min_lr, args.warmup_iters, args.max_iters)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        logits = model(X)
        loss = cross_entropy(logits, Y)
        loss.backward()

        gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        X, Y = get_batch(train_data, args.batch_size, args.max_seq_len, device)

        if iter_num % args.log_interval == 0:
            dt = time.time()
            to = time.time()
            loss_f = loss.item()

            print(f"iter {iter_num}: loss {loss_f:.4f}, time {dt*1000:.2f}ms, lr {lr:.6f}")

        if iter_num > 0 and iter_num % args.eval_interval == 0:
            print(f"evaluating at iter {iter_num}...")
            losses = estimate_loss(model, args.data_dir, args.batch_size, args.max_seq_len, device)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint_path = os.path.join(args.out_dir, 'ckpt_best.pt')
                    print(f"saving best checkpoint to {checkpoint_path}")
                    save_checkpoint(model, optimizer, iter_num, checkpoint_path)
        
        if iter_num > 0 and iter_num % args.save_interval == 0:
            checkpoint_path = os.path.join(args.out_dir, f'ckpt_{iter_num}.pt')
            print(f"saving regular checkpoint to {checkpoint_path}")
            save_checkpoint(model, optimizer, iter_num, checkpoint_path)

        iter_num += 1
    
    print(f"training completed")

if __name__ == "__main__":
    main()
    


