import datetime
import timeit


class Timer:

    def __init__(self):
        self.start_time = timeit.default_timer()

    def __call__(self) -> float:
        return timeit.default_timer() - self.start_time

    def __str__(self) -> str:
        return str(datetime.timedelta(seconds=round(self(), 0)))
