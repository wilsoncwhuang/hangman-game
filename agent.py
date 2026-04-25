import math
from collections import Counter
from typing import List

import numpy as np
import torch


def _next_prob(model, device, prefix_ids: List[int], cand_idx: List[int]) -> torch.Tensor:
    x = torch.tensor(prefix_ids, dtype=torch.long, device=device)[None, :]
    m = torch.ones_like(x, dtype=torch.bool)
    logits = model(x, attn_mask=m)[:, -1, :]
    prob = torch.softmax(logits, dim=-1).squeeze(0)
    return prob[cand_idx]


def _score_words_ar(words, tokenizer, lm, device):
    out = np.zeros(len(words), dtype=np.float64)
    for i, w in enumerate(words):
        ids = [tokenizer.bos_id] + [tokenizer.stoi[ch] for ch in w]
        lp = 0.0
        for t in range(1, len(ids)):
            p = _next_prob(lm, device, ids[:t], [ids[t]])[0].item()
            lp += math.log(max(p, 1e-12))
        out[i] = lp
    return out


def _letter_marginals(words, log_scores, pattern, missed, guessed, temperature=1.0):
    alphabet = [chr(c) for c in range(ord('a'), ord('z') + 1)]
    remaining = [a for a in alphabet if a not in missed | guessed]

    if len(words) == 0:
        return np.array([]), [(a, 0.0) for a in remaining]

    s = np.asarray(log_scores, dtype=np.float64) / max(1e-6, temperature)
    s -= s.max()
    wts = np.exp(s)
    post = wts / max(wts.sum(), 1e-12)

    hidden_idx = [i for i, ch in enumerate(pattern) if ch == '_']
    hit_p = {a: 0.0 for a in remaining}
    exp_cnt = {a: 0.0 for a in remaining}
    for w, pw in zip(words, post):
        for a in remaining:
            k = sum(1 for i in hidden_idx if w[i] == a)
            if k > 0:
                hit_p[a] += pw
                exp_cnt[a] += pw * k

    ranked = sorted(remaining, key=lambda a: (-hit_p[a], -exp_cnt[a], a))
    return post, [(a, hit_p[a]) for a in ranked]


@torch.inference_mode()
def _constrained_beam(pattern: str, missed, tokenizer, lm, beam_size: int,
                      max_candidates: int = 5000, temperature: float = 1.0):
    device = next(lm.parameters()).device
    L = len(pattern)
    alphabet = [chr(c) for c in range(ord('a'), ord('z') + 1)]
    cand_letters = [a for a in alphabet if a not in missed]
    cand_idx = [tokenizer.stoi[a] for a in cand_letters]
    goal = Counter(c for c in pattern if c != '_')

    def violates(prefix, counts):
        i = len(prefix) - 1
        ch = prefix[-1]
        if pattern[i] != '_' and pattern[i] != ch:
            return True
        if ch in goal and counts[ch] > goal[ch]:
            return True
        return False

    beams = [("", 0.0, Counter())]
    for _ in range(L):
        if not beams:
            break
        new_beams = []
        for s, lp, cnt in beams:
            prefix_ids = [tokenizer.bos_id] + [tokenizer.stoi[ch] for ch in s]
            probs = _next_prob(lm, device, prefix_ids, cand_idx)
            log_probs = (torch.log(probs + 1e-12) / max(1e-6, temperature))
            probs_np = torch.softmax(log_probs, dim=-1).cpu().numpy()
            for a, pa in zip(cand_letters, probs_np):
                cnt2 = cnt.copy()
                cnt2[a] += 1
                s2 = s + a
                if violates(s2, cnt2):
                    continue
                new_beams.append((s2, lp + math.log(max(pa, 1e-12)), cnt2))
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]

    cand_words = [
        (s, lp) for s, lp, cnt in beams
        if len(s) == L and all(cnt[c] == need for c, need in goal.items())
    ]
    if not cand_words:
        return [], np.array([], dtype=float)

    cand_words.sort(key=lambda x: x[1], reverse=True)
    cand_words = cand_words[:max_candidates]
    words, scores = zip(*cand_words)
    return list(words), np.array(scores, dtype=float)


@torch.inference_mode()
def guess(pattern: str, guessed_letters: List[str],
          fwd_tokenizer, bwd_tokenizer, fwd_lm, bwd_lm,
          bidirection: bool = True, beam_size: int = 128, lam: float = 0.5) -> str:
    guessed = set(guessed_letters)
    present = {ch for ch in pattern if ch != '_'}
    missed = {g for g in guessed if g not in present}
    device = next(fwd_lm.parameters()).device

    fwd_words, fwd_scores = _constrained_beam(pattern, missed, fwd_tokenizer, fwd_lm, beam_size)

    if bidirection:
        bwd_words_rev, _ = _constrained_beam(pattern[::-1], missed, bwd_tokenizer, bwd_lm, beam_size)
        bwd_words = [w[::-1] for w in bwd_words_rev]
        words = list(dict.fromkeys(fwd_words + bwd_words))
        fwd_s = _score_words_ar(words, fwd_tokenizer, fwd_lm, device)
        bwd_s = _score_words_ar([w[::-1] for w in words], bwd_tokenizer, bwd_lm, device)
        scores = lam * fwd_s + (1 - lam) * bwd_s
    else:
        words, scores = fwd_words, fwd_scores

    _, ranked = _letter_marginals(words, scores, pattern, missed, guessed)
    return ranked[0][0]
