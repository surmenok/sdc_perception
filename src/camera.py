# --------------------------------------------------------
# Reading images from camera for Tegra X2/X1
#
# Code to capture video is based on code by JK Jung <jkjung13@gmail.com>
# Refer to the following blog post for details of the original implementation:
#   https://jkjung-avt.github.io/tx2-camera-with-python/
#
# Code of Camera class is based mostly on code by Adrian Rosebrock
# Source: https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
# --------------------------------------------------------

import sys
import cv2
import numpy as np
from threading import Thread, Lock

class Camera:
    def __init__(self):
        self._lock = Lock()
        self._stream = self._open_stream()
        with self._lock:
              (self.grabbed, self.frame) = self._stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self._update, args=(), daemon=True).start()
        return self

    def read(self):
        with self._lock:
            return np.roll(self.frame, 1, axis=-1)

    def stop(self):
        self.stopped = True

    def close(self):
        self._stream.release()

    def _open_stream(self):
        raise NotImplementedError()

    def _update(self):
        while True:
            if self.stopped:
                return
            with self._lock:
                (self.grabbed, self.frame) = self._stream.read()

class UsbCamera(Camera):
    def __init__(self, dev=1, width=1920, height=1080):
        """
        dev -> int: video device # of USB webcam (/dev/video?)
        width -> int: image width
        height -> int: image height
        """
        self.dev = dev
        self.width = width
        self.height = height
        super().__init__()

    def _open_stream(self):
        # We want to set width and height here, otherwise we could just do:
        #     return cv2.VideoCapture(dev)
        gst_str = ("v4l2src device=/dev/video{} ! "
                   "video/x-raw, width=(int){}, height=(int){}, format=(string)RGB ! "
                   "videoconvert ! appsink").format(self.dev, self.width, self.height)
        #return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
        return cv2.VideoCapture(self.dev)

class RtspCamera(Camera):
    def __init__(self, uri, width=1920, height=1080, latency=200):
        """
        uri -> str: RTSP URI string, e.g. rtsp://192.168.1.64:554
        width -> int: image width
        height -> int: image height
        latency -> int: latency in ms for RTSP
        """
        self.uri = uri
        self.width = width
        self.height = height
        self.latency = latency
        super().__init__()

    def _open_stream(self):
        gst_str = ("rtspsrc location={} latency={} ! rtph264depay ! h264parse ! omxh264dec ! "
                   "nvvidconv ! video/x-raw, width=(int){}, height=(int){}, format=(string)BGRx ! "
                   "videoconvert ! appsink").format(self.uri, self.latency, self.width, self.height)
        return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

class OnboardCamera(Camera):
    def __init__(self, width=1920, height=1080):
        """
        width -> int: image width
        height -> int: image height
        """
        self.width = width
        self.height = height
        super().__init__()

    def _open_stream(self):
        # On versions of L4T previous to L4T 28.1, flip-method=2
        # Use Jetson onboard camera
        gst_str = ("nvcamerasrc ! "
                   "video/x-raw(memory:NVMM), width=(int)2592, height=(int)1458, format=(string)I420, framerate=(fraction)30/1 ! "
                   "nvvidconv ! video/x-raw, width=(int){}, height=(int){}, format=(string)BGRx ! "
                   "videoconvert ! appsink").format(self.width, self.height)
        return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
