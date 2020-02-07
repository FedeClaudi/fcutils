import cv2
import os
import numpy as np


def get_cap_from_file(videopath):
    try:
        cap = cv2.VideoCapture(videopath)
    except Exception as e:
        raise ValueError("Could not open video at: " + videopath + f"\n {e}")

    ret, frame = cap.read()
    if not ret:
        raise ValueError("Something went wrong, can't read form video: " + videopath)

    return cap

def cap_set_frame(cap, frame_number):
    cap.set(1, frame_number)

def get_cap_selected_frame(cap, show_frame):
    cap_set_frame(cap, show_frame)
    ret, frame = cap.read()

    if not ret:
        return None
    else:
        return frame


def get_video_params(cap):
    if isinstance(cap, str):
        cap = cv2.VideoCapture(cap)

    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    return nframes, width, height, fps


def opencv_write_clip(
    videopath,
    frames_data,
    w=None,
    h=None,
    framerate=None,
    start=None,
    stop=None,
    format=".mp4",
    iscolor=False,
):
    """ create a .cv2 videowriter and  write clip to file """
    if format != ".mp4":
        raise ValueError(
            "Fileformat not yet supported by this function: {}".format(format)
        )

    if start is None:
        start = 0
    if stop is None:
        stop = frames_data.shape[-1]
    start, stop = int(start), int(stop)
    if w is None:
        w = frames_data.shape[0]
    if h is None:
        h = frames_data.shape[1]
    if framerate is None:
        raise ValueError("No frame rate parameter was given as an input")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    videowriter = cv2.VideoWriter(videopath, fourcc, framerate, (w, h), iscolor)

    for framen in tqdm(range(start, stop)):
        frame = np.array(frames_data[:, :, framen], dtype=np.uint8).T
        cv2.imshow("frame", frame)
        cv2.waitKey(1)
        videowriter.write(frame)
    videowriter.release()


def open_cvwriter(
    filepath, w=None, h=None, framerate=None, format=".mp4", iscolor=False
):
    try:
        if "avi" in format:
            fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")  # (*'MP4V')
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        videowriter = cv2.VideoWriter(filepath, fourcc, framerate, (w, h), iscolor)
    except:
        raise ValueError("Could not create videowriter")
    else:
        return videowriter


def play_video(videofilepath, faster=False, play_from=None, stop_after=None):
    import cv2

    cap = cv2.VideoCapture(videofilepath)

    if play_from is not None:
        cap.set(1, play_from)

    frame_counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("frame", frame)
        if faster:
            wait = 5
        else:
            wait = 15
        k = cv2.waitKey(wait)
        if k == ord("q"):
            break

        frame_counter += 1
        if stop_after is not None:
            if frame_counter >= stop_after:
                break


def manual_video_inspect(videofilepath):
    """[loads a video and lets the user select manually which frames to show]

            Arguments:
                    videofilepath {[str]} -- [path to video to be opened]

            key bindings:
                    - d: advance to next frame
                    - a: go back to previous frame
                    - s: select frame
                    - f: save frame
    """

    def get_selected_frame(cap, show_frame):
        cap.set(1, show_frame)
        ret, frame = cap.read()  # read the first frame
        return frame

    import cv2  # import opencv

    # Open text file to save selected frames
    fold, name = os.path.split(videofilepath)

    frames_file = open(os.path.join(fold, name.split(".")[0]) + ".txt", "w+")

    cap = cv2.VideoCapture(videofilepath)
    if not cap.isOpened():
        raise FileNotFoundError("Couldnt load the file")

    print(
        """ Instructions
                    - d: advance to next frame
                    - a: go back to previous frame
                    - s: select frame
                    - f: save frame number
                    - q: quit
    """
    )

    number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialise showing the first frame
    show_frame = 0
    frame = get_selected_frame(cap, show_frame)

    while True:
        cv2.imshow("frame", frame)

        k = cv2.waitKey(25)

        if k == ord("d"):
            # Display next frame
            if show_frame < number_of_frames:
                show_frame += 1
        elif k == ord("a"):
            # Display the previous frame
            if show_frame > 1:
                show_frame -= 1
        elif k == ord("s"):
            selected_frame = int(input("Enter frame number: "))
            if selected_frame > number_of_frames or selected_frame < 0:
                print(selected_frame, " is an invalid option")
            show_frame = int(selected_frame)
        elif k == ord("f"):
            print("Saving frame to text")
            frames_file.write("\n" + str(show_frame))
        elif k == ord("q"):
            frames_file.close()
            sys.exit()

        try:
            frame = get_selected_frame(cap, show_frame)
            print("Showing frame {} of {}".format(show_frame, number_of_frames))
        except:
            raise ValueError("Could not display frame ", show_frame)
