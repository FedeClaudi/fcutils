import cv2
from loguru import logger

from fcutils._video import open_video
from fcutils._file import pathify, raise_on_path_not_exists

# ------------------------ Create opencv video writers ----------------------- #


def open_cvwriter(
    filepath, w=None, h=None, framerate=None, format="mp4", iscolor=False
):
    """
        Creats an instance of cv.VideoWriter to write video to file

        Arguments:
            filepath: str, Path. Where video is to be saved
            w,h: int. width and height of frame in pixels
            framerate: int. fps of output video
            format: video format ('mp4' or 'avi')
            iscolor: bool, set as true if images are rgb, else false if they are gray

        Returns:
            cv2.VideoWriter
    """
    try:
        if "avi" in format:
            fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        videowriter = cv2.VideoWriter(
            filepath, fourcc, framerate, (w, h), iscolor
        )
    except:
        raise ValueError("Could not create videowriter")
    else:
        return videowriter


# --------------------- Manipulate opened video captures --------------------- #
def cap_set_frame(cap, frame_number):
    """
        Sets an opencv video capture object to a specific frame

        Arguments:
            cap: a VideoCapture object
            frame_number: int frame number
    """
    cap.set(1, frame_number)


def get_cap_selected_frame(cap, frame_number):
    """ 
        Gets a frame from an opencv video capture object to a specific frame

        Arguments:
            cap: a VideoCapture object
            frame_number: int frame number
    """
    cap_set_frame(cap, frame_number)
    ret, frame = cap.read()

    if not ret:
        logger.debug(f"FCUTILS: failed to read frame {frame_number}.")
        return None
    else:
        return frame


@open_video
def get_video_params(video):
    """ 
        Gets video parameters from an opencv video capture object
        
        Arguments:
            video: can be either the path to a video or a videocapture object

        Returns:
            nframes: int. Number of frames in video
            width, height: int. Frame size
            fps: int. Framerate
            is_color: bool. True if video si RGB false if it's 1 color.
    """
    frame = get_cap_selected_frame(video, 0)

    if frame is None:
        raise ValueError(
            "Could not read frame from cap while getting video params"
        )

    if frame.shape[1] == 3 or frame.shape[0] == 3:
        is_color = True
    else:
        is_color = False
    cap_set_frame(video, 0)

    nframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    return nframes, width, height, fps, is_color


# ---------------------------- open videocaptures ---------------------------- #


def get_cap_from_file(path):
    '''
        Opens a cv2 video capture from a file path
    '''
    try:
        return  cv2.VideoCapture(str(path))
    except Exception:
        logger.warning(f'Failed to open cap for video at: {path}')
        return None

@pathify
@raise_on_path_not_exists
def get_cap_from_images_folder(folder, img_format="%1d.png"):
    """
        Creates a video capture from a folder of images.

        Argiments:
            folder: str, path to folder with images
            img_format: str. Format of the file names
    """
    # Create video capture
    cap = cv2.VideoCapture(str(folder / img_format))

    # Check all went well
    ret, frame = cap.read()
    if not ret:
        raise ValueError(
            "Something went wrong, can't read form folder: " + folder
        )
    else:
        cap_set_frame(cap, 0)
    return cap


# --------------------------------- trim clip -------------------------------- #
@open_video
def trim_clip(
    video, savepath, start_frame=0, end_frame=-1, fps=None,
):
    """trim_clip [take a videopath, open it and save a trimmed version between start and stop. Either 
    looking at a proportion of video (e.g. second half) or at start and stop frames]
    
    Arguments:
        videopath {[str]} -- [video to process]
        savepath {[str]} -- [where to save]
    
    Keyword Arguments:
        frame_mode {bool} -- [define start and stop time as frame numbers] (default: {False})
        start_frame {int} -- [video frame to stat at ] (default: {None})
        end_frame {int} -- [videoframe to stop at ] (default: {None})
        fps {[int]}(default, None) -- [specify the fps of the output]
    """

    # Open input file and extract params
    nframes, width, height, _fps, is_color = get_video_params(video)
    fps = fps or _fps

    # open writer
    writer = open_cvwriter(
        str(savepath),
        w=width,
        h=height,
        framerate=int(fps),
        format=".mp4",
        iscolor=is_color,
    )

    # Loop over frames and save the ones that matter
    cap_set_frame(video, start_frame)
    for framen in range(start_frame, end_frame):
        ret, frame = video.read()
        if not ret:
            logger.debug(
                f"FCUTILS: failed to read frame {framen} while trimming clip [{nframes} frames long]."
            )
            break
        else:
            if not is_color:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            writer.write(frame)
    writer.release()
