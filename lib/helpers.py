import datetime
import timeit
from typing import List


class Timer:

    def __init__(self):
        self.start_time = timeit.default_timer()

    def __call__(self) -> float:
        return timeit.default_timer() - self.start_time

    def __str__(self) -> str:
        return str(datetime.timedelta(seconds=round(self(), 0)))


class Tokenizer:

    def __init__(self, vocabulary: List[str]):
        self._vocabulary = dict(zip(vocabulary, range(len(vocabulary))))

    def tokenize(self, sequence: List[str]) -> List[str]:
        return [self._vocabulary[key] for key in sequence]
