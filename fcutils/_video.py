import cv2
from pathlib import Path


def open_video(func):
    """
        Decorator. If the first argument to a function is a path
        to a video it returns the cv2.VideoCapture object for it
    """

    def inner(*args, **kwargs):
        if isinstance(args[0], (str, Path)):
            args = list(args)
            args[0] = cv2.VideoCapture(str(args[0]))
            return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    return inner
