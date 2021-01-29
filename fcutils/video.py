from fcutils._video import open_video


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
        return None
    else:
        return frame

@open_video
def get_video_params(video):
    """ 
        Gets video parameters from an opencv video capture object
        
        Arguments:
            video: can be either the path to a video or a videocapture object
    """
    frame = get_cap_selected_frame(video, 0)
    
    if frame is None:
        raise ValueError("Could not read frame from cap while getting video params")
    
    if frame.shape[1] == 3 or frame.shape[0] == 3:
        is_color = True
    else: is_color = False
    cap_set_frame(video, 0)

    nframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    return nframes, width, height, fps, is_color