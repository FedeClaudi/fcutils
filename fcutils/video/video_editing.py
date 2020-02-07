import sys

sys.path.append("./")

import warnings as warn

try:
    import cv2
except:
    pass
import os
from tempfile import mkdtemp
from tqdm import tqdm
from collections import namedtuple
from nptdms import TdmsFile
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
import shutil
import matplotlib.pyplot as plt
import time

from fcutils.file_io.io import *




class Editor:
    def __init__(self):
        raise NotImplementedError("This code is old and full of stuff that I broke")

    def trim_clip(
        self,
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
            selfpd {[int]}(default, None) -- [specify the fps of the output]
            lighten --> make the video a bit brighter
        """

        # Open reader and writer
        cap = cv2.VideoCapture(videopath)
        nframes, width, height, fps = self.get_video_params(cap)

        if sel_fps is not None:
            fps = sel_fps
        writer = self.open_cvwriter(
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
        print("Processing: ", videopath)
        cur_frame = 0
        cap.set(1, start_frame)
        while True:
            cur_frame += 1
            if cur_frame % 100 == 0:
                print("Current frame: ", cur_frame)
            if cur_frame <= start_frame:
                continue
            elif cur_frame >= stop_frame:
                break
            else:

                ret, frame = cap.read()
                if not ret:
                    break
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    if lighten:
                        a = 1
                    writer.write(frame)
        writer.release()

    def split_clip(self, clip, number_of_clips=4, dest_fld=None):
        """[Takes a video and splits into clips of equal length]
        
        Arguments:
            clip {[str]} -- [path to video to be split]
        
        Keyword Arguments:
            number_of_clips {int} -- [number of subclips] (default: {4})
            dest_fld {[srt]} -- [path to folder where clips will be saved. If None, clips will be saved in same folder as original clip] (default: {None})
        """

        fld, name = os.path.split(clip)
        name, ext = name.split(".")
        if dest_fld is None:
            dest_fld = fld

        cap = cv2.VideoCapture(clip)
        nframes, width, height, fps = self.get_video_params(cap)

        frames_array = np.linspace(0, nframes, nframes + 1)
        clips_frames = np.array_split(frames_array, number_of_clips)

        for i, clip in enumerate(clips_frames):

            start, end = clip[0], clip[-1]
            print(
                "Clip {} of {}, frame range: {}-{}".format(
                    i, number_of_clips, start, end
                )
            )
            if i == 0:
                print(" ... skipping the first clip")
                continue
            cap.set(1, start)

            savename = os.path.join(dest_fld, name + "_clip{}.".format(i) + ext)
            writer = self.open_cvwriter(
                savename, w=width, h=height, framerate=fps, iscolor=False
            )

            counter = start
            while counter <= end:
                counter += 1
                ret, frame = cap.read()
                if not ret:
                    writer.release()
                    return
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                writer.write(gray)
            writer.release()
        cap.release()

    def tile_clips(self, clips_l, savepath):
        """[Tiles multiple videos horizzontally. It assumes that all videos have the same width and height]
        
        Arguments:
            clips_l {[list]} -- [list of paths to the videos to be tiled]
            savepath {[type]} -- [complete filepath of the video to be saved]
        """
        caps = [cv2.VideoCapture(videofilepath) for videofilepath in clips_l]

        nframes, width, height, fps = self.get_video_params(caps[0])
        width *= len(caps)
        writer = self.open_cvwriter(
            savepath, w=width, h=height, framerate=fps, iscolor=True
        )

        while True:
            try:
                frames = [cap.read()[1] for cap in caps]
            except:
                break
            else:
                tot_frame = np.hstack(frames)
                writer.write(tot_frame)
        writer.release()

    def compress_clip(
        self, videopath, compress_factor, save_path=None, start_frame=0, stop_frame=None
    ):
        """
            takes the path to a video, opens it as opecv Cap and resizes to compress factor [0-1] and saves it
        """
        cap = cv2.VideoCapture(videopath)
        nframes, width, height, fps = self.get_video_params(cap)

        resized_width = int(np.ceil(width * compress_factor))
        resized_height = int(np.ceil(height * compress_factor))

        if save_path is None:
            save_name = (
                os.path.split(videopath)[-1].split(".")[0] + "_compressed" + ".mp4"
            )
            save_path = os.path.split(videopath)
            save_path = os.path.join(list(save_path))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        videowriter = self.open_cvwriter(
            save_path,
            w=resized_width,
            h=resized_height,
            framerate=fps,
            format=".mp4",
            iscolor=False,
        )
        framen = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (resized_width, resized_height))
            videowriter.write(resized)
            framen += 1

            if stop_frame is not None:
                if framen >= stop_frame:
                    break

        videowriter.release()

    def crop_video(self, videopath, x, y):
        cap = cv2.VideoCapture(videopath)
        nframes, width, height, fps = self.get_video_params(cap)

        path, name = os.path.split(videopath)
        name, ext = name.split(".")
        savename = os.path.join(path, name + "_cropped.mp4")

        writer = self.open_cvwriter(
            savename, w=x, h=y, framerate=fps, format=".mp4", iscolor=True
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cropped = frame[:y, :x, :]

            writer.write(cropped)
        writer.release()

    def concatenate_videos(self, videos):
        """[takes a list of paths as argument]
        """
        tot_frames = 0
        for video in videos:
            cap = cv2.VideoCapture(video)
            nframes, width, height, fps = self.get_video_params(cap)
            tot_frames += nframes

        path, name = os.path.split(video)
        name, ext = name.split(".")
        savename = os.path.join(path, name + "_concatenated.mp4")

        writer = self.open_cvwriter(
            savename, w=width, h=height, framerate=fps, format=".mp4", iscolor=True
        )

        for video in videos:
            cap = cv2.VideoCapture(video)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                writer.write(frame)
        writer.release()

    def brighten_video(self, videopath, save_path, add_value=100):
        cap = cv2.VideoCapture(videopath)
        nframes, width, height, fps = self.get_video_params(cap)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        videowriter = self.open_cvwriter(
            save_path, w=width, h=height, framerate=fps, format=".mp4", iscolor=False
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            gray = np.add(gray, add_value)
            gray[gray > 255] + 255
            videowriter.write(gray)
        videowriter.release()
