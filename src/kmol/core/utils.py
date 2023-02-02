from rich import progress as pb


def progress_bar():
    return pb.Progress(
        "[progress.description]{task.description}",
        pb.BarColumn(),
        pb.MofNCompleteColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        pb.TimeRemainingColumn(),
        pb.TimeElapsedColumn(),
    )
