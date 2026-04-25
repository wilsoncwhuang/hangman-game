import math
import torch
import torch.nn as nn
from typing import Iterable, List


class CharTokenizer:
    def __init__(self, extra_tokens: Iterable[str] = ("<pad>", "<bos>", "<eos>")):
        base = [chr(c) for c in range(ord('a'), ord('z') + 1)]
        self.tokens = list(extra_tokens) + base
        self.stoi = {c: i for i, c in enumerate(self.tokens)}
        self.itos = {i: t for t, i in self.stoi.items()}
        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]

    def encode(self, word: str, add_bos=True, add_eos=True) -> List[int]:
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        for c in word:
            if c not in self.stoi:
                continue
            ids.append(self.stoi[c])
        if add_eos:
            ids.append(self.eos_id)
        return ids


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class CharTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 4,
                 num_layers: int = 6, d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=64)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        h = self.token_emb(x)
        h = self.pos_enc(h)
        T = x.size(1)
        causal = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        h = self.encoder(
            h, mask=causal,
            src_key_padding_mask=(~attn_mask) if attn_mask is not None else None
        )
        h = self.ln(h)
        return self.out(h)


def load_lm(path: str):
    state = torch.load(path, map_location='cpu', weights_only=False)
    tokens = state['vocab']
    tokenizer = CharTokenizer()
    tokenizer.tokens = tokens
    tokenizer.stoi = {c: i for i, c in enumerate(tokens)}
    tokenizer.itos = {i: t for t, i in tokenizer.stoi.items()}
    tokenizer.pad_id = tokenizer.stoi["<pad>"]
    tokenizer.bos_id = tokenizer.stoi["<bos>"]
    tokenizer.eos_id = tokenizer.stoi["<eos>"]
    cfg = state['config']
    model = CharTransformerLM(
        vocab_size=len(tokenizer.tokens),
        d_model=cfg['d_model'], nhead=cfg['nhead'],
        num_layers=cfg['num_layers'], d_ff=cfg['d_ff'], dropout=cfg['dropout']
    )
    model.load_state_dict(state['model'])
    model.eval()
    return tokenizer, model
