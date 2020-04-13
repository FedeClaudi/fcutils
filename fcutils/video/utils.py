import cv2
import os
import numpy as np
from tqdm import tqdm




# ------------------------ Create opencv video writers ----------------------- #
def open_cvwriter(filepath, w=None, h=None, framerate=None, format=".mp4", iscolor=False):
    """
        Creats an instance of cv.VideoWriter to write frames to video using python opencv

        :param filepath: str, path to file to save
        :param w,h: width and height of frame in pixels
        :param framerate: fps of output video
        :param format: video format
        :param iscolor: bool, set as true if images are rgb, else false if they are gray
    """
    try:
        if "avi" in format:
            fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")  
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        videowriter = cv2.VideoWriter(filepath, fourcc, framerate, (w, h), iscolor)
    except:
        raise ValueError("Could not create videowriter")
    else:
        return videowriter

# --------------------- Manipulate opened video captures --------------------- #
def cap_set_frame(cap, frame_number):
    """
        Sets an opencv video capture object to a specific frame
    """
    cap.set(1, frame_number)

def get_cap_selected_frame(cap, show_frame):
    """ 
        Gets a frame from an opencv video capture object to a specific frame
    """
    cap_set_frame(cap, show_frame)
    ret, frame = cap.read()

    if not ret:
        return None
    else:
        return frame


def get_video_params(cap):
    """ 
        Gets video parameters from an opencv video capture object
    """
    if isinstance(cap, str):
        cap = cv2.VideoCapture(cap)

    frame = get_cap_selected_frame(cap, 0)
    if frame.shape[1] == 3:
        is_color = True
    else: is_color = False
    cap_set_frame(cap, 0)

    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    return nframes, width, height, fps, is_color


# --------------------------- Create video captures -------------------------- #

def get_cap_from_file(videopath):
    """
        Opens a video file as an opencv video capture
    """
    try:
        cap = cv2.VideoCapture(videopath)
    except Exception as e:
        raise ValueError("Could not open video at: " + videopath + f"\n {e}")

    ret, frame = cap.read()
    if not ret:
        raise ValueError("Something went wrong, can't read form video: " + videopath)
    else:
        cap_set_frame(cap, 0)
    return cap


def get_cap_from_images_folder(folder, img_format="%1d.png"):
    if not os.path.isdir(folder):
        raise ValueError(f"Folder {folder} doesn't exist")
    if not os.listdir(folder):
        raise ValueError(f"Folder {folder} is empty")

    # Create video capture
    cap = cv2.VideoCapture(os.path.join(folder, img_format))

    # Check all went well
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Something went wrong, can't read form folder: " + folder)
    else:
        cap_set_frame(cap, 0)
    return cap




# ---------------------- Create video from video capture --------------------- #
def save_videocap_to_video(cap, savepath, fmt, fps=30, iscolor=True):
    """
        Saves the content of a videocapture opencv object to a file
    """
    if "." not in fmt: fmt = "."+fmt
    # Creat video writer
    nframes, width, height, _, _ = get_video_params(cap)
    writer = open_cvwriter(savepath, w=width, h=height, framerate=fps, format=fmt, iscolor=iscolor)

    # Save frames
    while True:
        ret, frame = cap.read()
        if not ret: break

        writer.write(frame)

    # Release everything if job is finished
    cap.release()
    writer.release()








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




def trim_clip(
    videopath,
    savepath,
    frame_mode=False,
    start=0.0,
    stop=0.0,
    start_frame=None,
    stop_frame=None,
    sel_fps=None,
    lighten=False,
):
    """trim_clip [take a videopath, open it and save a trimmed version between start and stop. Either 
    looking at a proportion of video (e.g. second half) or at start and stop frames]
    
    Arguments:
        videopath {[str]} -- [video to process]
        savepath {[str]} -- [where to save]
    
    Keyword Arguments:
        frame_mode {bool} -- [define start and stop time as frame numbers] (default: {False})
        start {float} -- [video proportion to start at ] (default: {0.0})
        end {float} -- [video proportion to stop at ] (default: {0.0})
        start_frame {[type]} -- [video frame to stat at ] (default: {None})
        end_frame {[type]} -- [videoframe to stop at ] (default: {None})
        sel_fps {[int]}(default, None) -- [specify the fps of the output]
        lighten --> make the video a bit brighter
    """

    # Open reader and writer
    cap = cv2.VideoCapture(videopath)
    nframes, width, height, fps = get_video_params(cap)

    if sel_fps is not None:
        fps = sel_fps
    writer = open_cvwriter(
        savepath,
        w=width,
        h=height,
        framerate=int(fps),
        format=".mp4",
        iscolor=False,
    )

    # if in proportion mode get start and stop mode
    if not frame_mode:
        start_frame = int(round(nframes * start))
        stop_frame = int(round(nframes * stop))

    # Loop over frames and save the ones that matter
    print("Processing: ", videopath, f'{stop_frame-start_frame} frames ')
    cur_frame = 0
    cap.set(1, start_frame)
    while True:
        cur_frame += 1

        if cur_frame <= start_frame:
            continue
        elif cur_frame >= stop_frame:
            break
        else:
            if cur_frame % 5000 == 0:
                print("Current frame: ", cur_frame)
                
            ret, frame = cap.read()
            if not ret:
                break
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if lighten:
                    a = 1
                writer.write(frame)
    writer.release()