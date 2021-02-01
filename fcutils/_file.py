from pathlib import Path


def sizeof_fmt(num, suffix="B"):
    """
        Returns a size in bytes to a human
        readable format
    """
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z", "Y"]:
        if abs(num) < 1024.0:
            return f"{num:.2f} {unit}{suffix}"
        num /= 1024.0


def raise_on_path_not_exists(func):
    """
        Decorator that raises an error when the
        path passed to a function does not point towards an actual file
    """

    def inner(*args, **kwargs):
        args = list(args)
        path = args[0]
        if not path.exists():
            raise FileNotFoundError(f"File or folder: {path} does not exist")
        else:
            return func(*args, **kwargs)

    return inner


def pathify(func):
    """ 
        Decorator that makes sure that the first arugment to a function
        is a Path object and not a path as string
    """

    def inner(*args, **kwargs):
        args = list(args)
        args[0] = Path(args[0])
        return func(*args, **kwargs)

    return inner


def return_list_smart(func):
    """ 
        Decorator. If a function is trying to return a list
        of length 1 it returns the list element instead
    """

    def inner(*args, **kwargs):
        out = func(*args, **kwargs)
        if not out:
            return None

        if len(out) == 1:
            return out[0]
        else:
            return out

    return inner
