from datetime import datetime


def timestamp(just_time=False):
    if not just_time:
        return datetime.now().strftime("%Y-%m-%d  %H:%M")
    else:
        return datetime.now().strftime("%H:%M:%S")


if __name__ == "__main__":
    print(timestamp())
