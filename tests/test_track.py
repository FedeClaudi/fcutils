from fcutils.progress import track, progress
import numpy as np


def test_track():
    N = np.linspace(0, 1000, 1)

    for n in track(np.arange(N), transient=True):
        pass


def test_progress():
    N = np.linspace(0, 1000, 1)
    tid = progress.add_task("count")

    with progress:
        for n in np.arange(N):
            progress.update(tid, completed=n)
            pass
