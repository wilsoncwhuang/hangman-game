import argparse
import math
import os
import random
from dataclasses import dataclass
from pprint import pprint
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import CharTokenizer, CharTransformerLM


def load_word_list(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        return [w.strip().lower() for w in f if w.strip()]


def split_words(words: List[str], ratio=(0.9, 0.1), seed: int = 42):
    rng = random.Random(seed)
    rng.shuffle(words)
    n_train = int(ratio[0] * len(words))
    return words[:n_train], words[n_train:]


class WordDataset(Dataset):
    def __init__(self, words: List[str], tokenizer: CharTokenizer, max_len: int = 32, reverse: bool = False):
        valid = set(tokenizer.stoi) - {"<pad>", "<bos>", "<eos>"}
        self.words = [w for w in words if all(c in valid for c in w) and 1 <= len(w) <= max_len]
        self.tokenizer = tokenizer
        self.reverse = reverse

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        w = self.words[idx]
        if self.reverse:
            w = w[::-1]
        ids = self.tokenizer.encode(w)
        return torch.tensor(ids[:-1], dtype=torch.long), torch.tensor(ids[1:], dtype=torch.long)


def collate_pad(batch, pad_id: int):
    xs, ys = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    xpad = torch.full((len(xs), max_len), pad_id, dtype=torch.long)
    ypad = torch.full((len(xs), max_len), pad_id, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        xpad[i, :x.size(0)] = x
        ypad[i, :y.size(0)] = y
    return xpad, ypad, xpad != pad_id


@dataclass
class TrainConfig:
    batch_size: int = 256
    epochs: int = 128
    lr: float = 1e-3
    grad_clip: float = 1.0
    weight_decay: float = 0.01
    max_len: int = 32
    gpu_id: int = 0


def train(train_words, val_words, output_path: str,
          d_model=256, nhead=4, num_layers=6, d_ff=1024, dropout=0.1,
          config: TrainConfig = TrainConfig(), reverse: bool = False):

    log_path = os.path.splitext(output_path)[0] + '_log.txt'
    tokenizer = CharTokenizer()
    train_ds = WordDataset(train_words, tokenizer, config.max_len, reverse)
    val_ds = WordDataset(val_words, tokenizer, config.max_len, reverse)
    collate = lambda b: collate_pad(b, tokenizer.pad_id)
    train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate)

    device = torch.device(f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu')
    model = CharTransformerLM(len(tokenizer.tokens), d_model, nhead, num_layers, d_ff, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    with open(log_path, 'w', encoding='utf-8') as log_f:
        log_f.write("epoch | train_loss | train_ppl | valid_loss | valid_ppl\n")
        for epoch in range(config.epochs):
            model.train()
            train_loss = train_tokens = 0
            for x, y, mask in tqdm(train_dl, desc=f"Epoch {epoch+1}/{config.epochs}", leave=False, ncols=80):
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x, attn_mask=mask)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                n = mask.sum().item()
                train_loss += loss.item() * n
                train_tokens += n
            train_ppl = math.exp(train_loss / max(train_tokens, 1))

            model.eval()
            val_loss = val_tokens = 0
            with torch.no_grad():
                for x, y, mask in val_dl:
                    x, y, mask = x.to(device), y.to(device), mask.to(device)
                    logits = model(x, attn_mask=mask)
                    loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                    n = mask.sum().item()
                    val_loss += loss.item() * n
                    val_tokens += n
            val_ppl = math.exp(val_loss / max(val_tokens, 1))

            log_f.write(f"{epoch+1} | {train_loss/train_tokens:.6f} | {train_ppl:.4f} | "
                        f"{val_loss/val_tokens:.6f} | {val_ppl:.4f}\n")
            log_f.flush()

    torch.save({
        'model': model.state_dict(),
        'vocab': tokenizer.tokens,
        'config': {'d_model': d_model, 'nhead': nhead, 'num_layers': num_layers, 'd_ff': d_ff, 'dropout': dropout},
    }, output_path)
    print(f"Saved model to {output_path}")


def main(args):
    config = TrainConfig(batch_size=args.batch, epochs=args.epochs, max_len=args.max_len, gpu_id=args.gpu_id)
    pprint(vars(config))

    words = load_word_list(args.words)
    for run in range(args.runs):
        train_words, val_words = split_words(words, seed=run)
        suffix = 'bwd' if args.reverse else 'fwd'
        out = os.path.join(args.out_dir, f"lm_{suffix}_{run + 1}.pt")
        train(train_words, val_words, out, args.d_model, args.nhead, args.num_layers, args.d_ff,
              args.dropout, config, args.reverse)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Train a character-level Transformer LM for Hangman")
    p.add_argument('--words', default='words_250000_train.txt', help='Training word list')
    p.add_argument('--out_dir', default='.', help='Directory to save model checkpoints')
    p.add_argument('--d_model', type=int, default=256)
    p.add_argument('--nhead', type=int, default=4)
    p.add_argument('--num_layers', type=int, default=6)
    p.add_argument('--d_ff', type=int, default=1024)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--reverse', action='store_true', help='Train backward LM (reversed words)')
    p.add_argument('--epochs', type=int, default=128)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--max_len', type=int, default=32)
    p.add_argument('--runs', type=int, default=1)
    p.add_argument('--gpu_id', type=int, default=0)
    main(p.parse_args())
