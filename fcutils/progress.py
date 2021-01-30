from rich.progress import (
    Progress,
    BarColumn,
    DownloadColumn,
    TextColumn,
    TransferSpeedColumn,
    TimeRemainingColumn,
)

from myterial import (
    orange,
    amber_light,
)


class TimeRemainingColumn(ProgressColumn):
    """Renders estimated time remaining."""

    # Only refresh twice a second to prevent jitter
    max_refresh = 0.5

    def render(self, task):
        """Show time remaining."""
        remaining = task.time_remaining
        if remaining is None:
            return Text("-:--:--", style=teal_light)
        remaining_delta = timedelta(seconds=int(remaining))
        return Text("remaining: " + str(remaining_delta), style=teal_light)


class TimeElapsedColumn(ProgressColumn):
    """Renders estimated time elapsed."""

    # Only refresh twice a second to prevent jitter
    max_refresh = 0.5

    def render(self, task):
        """Show time elapsed."""
        elapsed = task.elapsed
        if elapsed is None:
            return Text("-:--:--", style=light_blue_light)
        elapsed_delta = timedelta(seconds=int(elapsed))
        return Text("elapsed: " + str(elapsed_delta), style=light_blue_light)


class SpeedColumn(TextColumn):
    _renderable_cache = {}

    def __init__(self, *args):
        pass

    def render(self, task):
        if task.speed is None:
            return " "
        else:
            return f"{task.speed:.1f} steps/s"


progress = Progress(
        description,
        BarColumn(bar_width=None),
        'Completed: ',
        TextColumn("[bold magenta]Completed {task.completed}/{task.total}"),
        "([progress.percentage]{task.percentage:>3.0f}%)",
        'Speed: ',
        SpeedColumn(),
        "•",
        'Remaining: ',
        TimeRemainingColumn(),
        'Elpsed: ',
        TimeElapsedColumn(),
        transient=False,
    )



def track(iterable, total=None, description='Working...'):
    '''
        Spawns a progress bar to monitor the progress of a for loop over
        an iterable sequence with detailed information.

        Arguments:
            iterable: list or other iterable object
            total: int. Total length of iterable
            description: str. Text to preprend to the progress bar.

        Returs:
            elements of iterable
    '''
    description = f'[{orange}]' + description

    if total is None:
        try:
            total = len(iterable)
        except Exception:
            raise ValueError('Could not get total from iterable, pass a total value.')

    with progress:
        yield from progress.track(
            iterable, total=total, description=description,
        )