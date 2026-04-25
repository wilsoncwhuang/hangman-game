import argparse
import random
import torch
from tqdm import tqdm

from model import load_lm
from agent import guess as agent_guess
from game import HangmanGame


def load_test_words(n: int, min_len: int = 3, max_len: int = 15, seed: int = 42) -> list:
    try:
        import nltk
        try:
            nltk.data.find('corpora/words')
        except LookupError:
            print("Downloading NLTK words corpus...")
            nltk.download('words', quiet=True)
        from nltk.corpus import words as nltk_words
        pool = [w.lower() for w in nltk_words.words()
                if w.isalpha() and min_len <= len(w) <= max_len]
    except ImportError:
        raise ImportError("nltk is required: pip install nltk")

    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:n]


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading models...")
    fwd_tokenizer, fwd_lm = load_lm(args.fwd_model)
    bwd_tokenizer, bwd_lm = load_lm(args.bwd_model)
    fwd_lm.to(device)
    bwd_lm.to(device)

    print(f"Loading {args.n_words} test words from NLTK...")
    test_words = load_test_words(args.n_words)
    print(f"Test vocabulary: {len(test_words)} words")

    game = HangmanGame(test_words, max_wrong=args.max_wrong)

    wins = 0
    for i in tqdm(range(args.games), desc="Playing", disable=args.verbose):
        pattern = game.new_game()
        guessed = []
        status = 'ongoing'

        if args.verbose:
            print(f"\n{'='*40}")
            print(f"Word: {' '.join(pattern)}")

        while status == 'ongoing':
            letter = agent_guess(
                pattern, guessed,
                fwd_tokenizer, bwd_tokenizer, fwd_lm, bwd_lm,
                bidirection=not args.fwd_only,
                beam_size=args.beam_size,
                lam=args.lam,
            )
            guessed.append(letter)
            status, pattern, wrong = game.guess(letter)
            tries_left = args.max_wrong - wrong

            if args.verbose:
                hit = letter in game.secret
                result = 'HIT ' if hit else 'MISS'
                print(f"  Guess: '{letter}'  [{result}]  {' '.join(pattern)}  "
                      f"(tries left: {tries_left})")

        if status == 'success':
            wins += 1
        if args.verbose:
            outcome = 'WIN' if status == 'success' else 'LOSE'
            print(f"  → {outcome}  answer: {game.secret}")

    print(f"\nResults: {wins}/{args.games} wins — success rate = {wins / args.games:.3f}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Play Hangman with a trained character LM")
    p.add_argument('--fwd_model', default='lm_fwd_1.pt', help='Path to forward LM checkpoint')
    p.add_argument('--bwd_model', default='lm_bwd_1.pt', help='Path to backward LM checkpoint')
    p.add_argument('--games', type=int, default=200, help='Number of games to play')
    p.add_argument('--n_words', type=int, default=2000, help='Test vocabulary size (sampled from NLTK)')
    p.add_argument('--max_wrong', type=int, default=6, help='Max wrong guesses before losing')
    p.add_argument('--beam_size', type=int, default=128, help='Beam size for constrained search')
    p.add_argument('--lam', type=float, default=0.5, help='Forward/backward blend weight')
    p.add_argument('--fwd_only', action='store_true', help='Use forward LM only (no bidirectional)')
    p.add_argument('--verbose', action='store_true', help='Print each game result')
    main(p.parse_args())
