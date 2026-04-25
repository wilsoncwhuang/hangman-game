# Hangman with Bidirectional Character Language Models

Automated Hangman player using Transformer-based character language models with constrained beam search and Bayesian posterior letter selection.

**Achieved 76% win rate** on 1,000 held-out games (vs. ~18% for a frequency-counting baseline).

---

## Introduction

The Hangman game is a sequential word-guessing problem that requires balancing exploring new letters and selecting letters that maximize the likelihood of completing the target word. This project investigates progressively more advanced approaches to improve the success rate in automated Hangman play.

---

## Method

### 1. Forward LM Scoring

A Transformer-based forward language model trained with a next-character prediction objective. The model predicts the probability distribution $P_{\text{fwd}}(c_t \mid c_{<t})$ for the next character given the current prefix, so the full word probability is:

$P_{\text{fwd}}(w) = \prod_{t=1}^{|w|} P_{\text{fwd}}(c_t \mid c_{<t})$

The letter with the highest softmax probability among unguessed letters is chosen at each step.

### 2. Beam Search with Posterior Scoring

The forward LM optimizes local next-character likelihoods $P(c_t \mid c_{<t})$ rather than the Hangman objective $P(a \in w \mid X)$. To address this, constrained beam search generates candidate words consistent with the current game state (revealed letters, known misses, word length), and their cumulative log-likelihoods are converted into a posterior over words:

$$s_{\text{beam}}(w) = \sum_{t=1}^{|w|} \log P_{\text{beam}}(c_t \mid c_{<t})$$

$$P(w \mid X) = \frac{P(X \mid w)\exp(s_{\text{beam}}(w)/T)}{\sum_{w'} P(X \mid w')\exp(s_{\text{beam}}(w')/T)}$$

Because the beam search enforces strict length, position, and letter constraints, each surviving candidate is almost perfectly consistent with the observation, so $P(X \mid w) \approx 1$. From this posterior, the marginal hit probability and expected number of reveals for each letter $a$ are:

$$P(a \in w \mid X) = \sum_{w:\, a \in w} P(w \mid X), \qquad \mathbb{E}[\text{reveal}(a) \mid X] = \sum_{w} P(w \mid X)\, \text{count}_w(a)$$

The letter maximizing this posterior utility is selected, leveraging global word-level information rather than local next-character likelihoods.

### 3. Bidirectional Autoregressive Scoring

To capture both left-to-right and right-to-left context, forward and backward LMs generate two candidate sets independently. Each candidate word is then re-scored by computing its exact autoregressive log-likelihood under both models:

$$s_{\text{fwd}}(w) = \sum_{t=1}^{\lvert w \rvert} \log P_{\text{fwd}}(c_t \mid c_{<t}), \qquad s_{\text{bwd}}(w) = \sum_{t=1}^{\lvert w \rvert} \log P_\text{bwd}(c_t \mid c_{>t})$$

The two scores are blended into a single bidirectional score:

$$s_{\text{bi}}(w) = \alpha\, s_{\text{fwd}}(w) + (1-\alpha)\, s_{\text{bwd}}(w)$$

where $\alpha \in [0, 1]$ controls the weighting between the two directions. The combined score is normalized into a posterior:

$$P_{\text{bi}}(w \mid X) = \frac{\exp(s_{\text{bi}}(w)/T)}{\sum_{w'} \exp(s_{\text{bi}}(w')/T)}$$

---

## Results

| Model | Beam Size | Success Rate |
|---|---|---|
| Baseline (frequency counting) | — | 18% |
| Forward LM Scoring | — | 45.2% |
| Posterior Beam Search | 128 | 67.5% |
| Bidirectional AR Scoring | 64+64 | 72% |
| Bidirectional AR Scoring | 128+128 | **77.5%** |

*Evaluated on 200 practice games. Final submission used beam size 256 (128+128) with $\alpha=0.5$, achieving **76%** on 1,000 recorded games.*

---

## Project Structure

```
model.py     — CharTokenizer, CharTransformerLM, load_lm
agent.py     — constrained beam search and guessing logic
game.py      — local Hangman game engine
train.py     — training script
play.py      — run N games and report win rate
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Training

Train the forward and backward LMs on your word corpus:

```bash
# Forward LM
python train.py --words path/to/words.txt --out_dir .

# Backward LM
python train.py --words path/to/words.txt --out_dir . --reverse
```

| Argument | Default | Description |
|---|---|---|
| `--words` | `words_250000_train.txt` | Path to training word list (one word per line) |
| `--out_dir` | `.` | Directory to save `.pt` checkpoints |
| `--epochs` | 128 | Training epochs |
| `--batch` | 256 | Batch size |
| `--d_model` | 256 | Transformer hidden size |
| `--num_layers` | 6 | Number of Transformer layers |
| `--reverse` | off | Train backward LM |

---

## Playing

```bash
python play.py --fwd_model lm_fwd_1.pt --bwd_model lm_bwd_1.pt --games 200
```

Test words are sampled from the [NLTK words corpus](https://www.nltk.org/), a public dictionary separate from the training data.

| Argument | Default | Description |
|---|---|---|
| `--games` | 200 | Number of games to play |
| `--n_words` | 2000 | Test vocabulary size (sampled from NLTK) |
| `--beam_size` | 128 | Beam size for constrained search |
| `--lam` | 0.5 | Forward/backward blend weight $\alpha$ |
| `--fwd_only` | off | Use forward LM only |
| `--verbose` | off | Print each game result |
