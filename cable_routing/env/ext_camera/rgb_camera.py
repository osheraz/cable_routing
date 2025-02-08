import cv2
import time
import threading
from queue import Queue
import gc
import os
import numpy as np
import subprocess


class Camera:
    def __init__(self, camera_index=0, fps=30):
        # Initialize the camera
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        self.set_camera_properties(fps)
        self.camera_index = camera_index

    def set_camera_properties(self, fps):
        # Set video resolution to 1920x1080 (MJPG)
        # 2K: 2560x1440
        # 4K: 4096x2160
        cam = self.cap
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 4096)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        cam.set(cv2.CAP_PROP_FPS, fps)
        # Brightness (0-255)
        cam.set(cv2.CAP_PROP_BRIGHTNESS, 128)

        # Contrast (0-255)
        cam.set(cv2.CAP_PROP_CONTRAST, 128)

        # Saturation (0-255)
        cam.set(cv2.CAP_PROP_SATURATION, 128)

        # Gain (0-100)
        cam.set(cv2.CAP_PROP_GAIN, 70)#128)

        # White Balance - Auto
        cam.set(cv2.CAP_PROP_AUTO_WB, 1)

        # Backlight Compensation
        cam.set(cv2.CAP_PROP_BACKLIGHT, 0)

        # Exposure (0.75 = Aperture Priority Mode)
        # Set exposure time priority
        # 1 means manual exposure mode
        cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 0.25 for exposure time priority mode

        # Set a specific exposure value (this value depends on your camera)
        cam.set(cv2.CAP_PROP_EXPOSURE, -7)

        # Disable Auto Focus and set manual focus
        cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        # cam.set(cv2.CAP_PROP_FOCUS, 30)

        cam.set(cv2.CAP_PROP_ZOOM, 100)

    def capture_image(self):
        # Capture a single frame
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to capture image")
        gc.collect()
        return frame

    def release(self):
        # Release the camera
        self.cap.release()

    def set_focus(self, focus):
        # Set manual focus to FOCUS
        self.cap.set(cv2.CAP_PROP_FOCUS, focus)

    def set_gain(self, gain):
        # Set manual gain to GAIN
        self.cap.set(cv2.CAP_PROP_GAIN, gain)
    
    def set_exposure(self, exposure):
        # Set manual exposure to EXPOSURE
        self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    
    def disable_exposure_dynamic_framerate(self):
        """
        Turn off 'exposure_dynamic_framerate' for the camera at /dev/video<camera_index>.
        """
        cmd = [
            "v4l2-ctl",
            f"--device=/dev/video{self.camera_index}",
            "--set-ctrl=exposure_dynamic_framerate=0"
        ]
        subprocess.run(cmd, check=True)


class CameraThread:
    def __init__(
        self,
        camera_index=0,
        fps=30,
        focus_queue=None,
    ):
        self.fps = fps
        self.camera = Camera(camera_index, fps)
        self.stop_thread = False
        self.frame = None
        self.interface = interface
        self.data_queue = Queue()
        self.focus_queue = focus_queue
        self.thread = threading.Thread(target=self.update, daemon=True)

    def start(self):
        """Start the video capture thread."""
        self.thread.start()

    def update(self):
        """Continuously capture frames in the background."""
        while not self.stop_thread:
            time.sleep(1 / 100)
            # If there's a focus command, set the camera focus
            if self.focus_queue is not None and not self.focus_queue.empty():
                focus = self.focus_queue.pop()  # .get()
                self.camera.set_focus(focus)

            # Capture a frame
            frame = self.camera.capture_image()

            # If the capture is successful...
            if frame is not None:
                self.frame = frame  # always keep the latest frame

            else:
                print("Failed to return frame")

    def get_frame(self):
        """Get the latest captured frame."""
        return self.frame

    def stop(self):
        """Stop the video capture thread."""
        self.stop_thread = True
        self.thread.join()
        self.camera.release()

    def change_focus(self, focus):
        self.camera.set_focus(focus)
    
    def change_gain(self, gain):
        self.camera.set_gain(gain)

    def change_exposure(self, exposure):
        self.camera.set_exposure(exposure)

    def disable_exposure_dynamic_framerate(self):
        """
        Turn off 'exposure_dynamic_framerate' for the camera at /dev/video<camera_index>.
        """
        self.camera.disable_exposure_dynamic_framerate()