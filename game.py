import random
from typing import List, Optional, Tuple


class HangmanGame:
    def __init__(self, word_list: List[str], max_wrong: int = 6):
        self.word_list = [w.lower() for w in word_list if w.isalpha()]
        self.max_wrong = max_wrong
        self.secret = ""
        self.guessed: set = set()
        self.wrong = 0

    def new_game(self, word: Optional[str] = None) -> str:
        self.secret = (word or random.choice(self.word_list)).lower()
        self.guessed = set()
        self.wrong = 0
        return self._pattern()

    def guess(self, letter: str) -> Tuple[str, str, int]:
        """Returns (status, pattern, wrong_count). Status: 'success'|'failed'|'ongoing'."""
        letter = letter.lower()
        self.guessed.add(letter)
        if letter not in self.secret:
            self.wrong += 1
        pattern = self._pattern()
        if '_' not in pattern:
            return 'success', pattern, self.wrong
        if self.wrong >= self.max_wrong:
            return 'failed', pattern, self.wrong
        return 'ongoing', pattern, self.wrong

    def _pattern(self) -> str:
        return ''.join(c if c in self.guessed else '_' for c in self.secret)

    @property
    def tries_remaining(self) -> int:
        return self.max_wrong - self.wrong
