import logging
import numpy as np
import matplotlib.pyplot as plt
from autolab_core import CameraIntrinsics
from cable_routing.configs.envconfig import BrioConfig
import cv2
import time
import os

logger = logging.getLogger(__name__)

class SensorIOException(Exception):
    pass

class BRIOSensor():
    """A driver for BRIO.

    Parameters
    ----------
    device : int
        v4l2 port (/dev/video<#> number) for the BRIO camera to connect to.
    """

    def __init__(self, device, inpaint=False, video=True, crop=(0, 0, 0, 0)):
        self._device = int(device)
        self._is_running = False
        self._crop = crop
        self._inpaint = inpaint
        self.video = video

        # Load configuration
        self.brio_config = BrioConfig()
        self.width = self.brio_config.width
        self.height = self.brio_config.height
        self.width_ = self.brio_config.width_
        self.height_ = self.brio_config.height_
        self.BRIO_DIST = np.asarray(self.brio_config.BRIO_DIST)

        # Camera Intrinsics
        self._camera_intr = self.create_intr(
            self.width,
            self.height,
            self.brio_config.fx,
            self.brio_config.fy,
            self.brio_config.cx,
            self.brio_config.cy
        )

        # Optimal new camera matrix
        self.newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(
            self._camera_intr.K, self.BRIO_DIST, (self.width, self.height), 1, (self.width, self.height)
        )

        # Undistortion mapping
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(
            self._camera_intr.K, self.BRIO_DIST, None, self.newcameramtx, (self.width, self.height), 5
        )

        # Initialize camera
        self.initialize()

    def initialize(self):
        """Initialize the BRIO camera."""
        # Check if device exists
        if not any(f.startswith(f"video{self._device}") for f in os.listdir('/dev')):
            raise SensorIOException(f"Device {self._device} not found; check `ls /dev/video*`")

        # Open camera
        self.cap = cv2.VideoCapture(self._device)
        if not self.cap.isOpened():
            raise SensorIOException(f"Failed to open BRIO camera on /dev/video{self._device}")

        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Turn autofocus off
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FPS, self.brio_config.FPS)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not self.video:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if np.any(self._crop):
            self._camera_intr = self._camera_intr.crop(*self._crop)

        # Verify camera feed
        ret, _ = self.cap.read()
        self._is_running = ret

        if not ret:
            logger.error("Failed to initialize BRIO camera, retrying...")
            self.cap.release()

    @staticmethod
    def create_intr(width, height, fx, fy, cx, cy):
        """Create camera intrinsics."""
        return CameraIntrinsics(
            fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height, frame='brio'
        )

    @property
    def device_name(self):
        """Get the device number."""
        return self._device

    @property
    def is_running(self):
        """Check if the stream is running."""
        return self._is_running

    @property
    def intrinsics(self):
        """Get the camera intrinsics."""
        return self._camera_intr

    @intrinsics.setter
    def intrinsics(self, new_in):
        """Set new intrinsics."""
        self._camera_intr = new_in

    @property
    def ir_intrinsics(self):
        """Get infrared camera intrinsics."""
        return self._camera_intr

    @property
    def ir_frame(self):
        """Get infrared frame."""
        return self.intrinsics.frame

    @property 
    def frame(self):
        """Get current frame."""
        return self.intrinsics.frame

    def frames(self):
        """Get frames for CameraChessboardRegistration."""
        return self.read()

    def read(self):
        """Read data from the sensor and return an RGB image."""
        self._color_im = None
        ret, frame = self.cap.read()

        if not ret:
            logger.warning("Re-initializing BRIO camera...")
            self.cap.release()
            self._is_running = False
            cv2.destroyAllWindows()
            self.initialize()

            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to read frame after re-initialization.")
                return None

        # Undistort and crop
        dst = cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR)
        x, y, w, h = self.roi
        color = dst[y:y+h, x:x+w]  # Crop ROI
        return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

    def start(self):
        """Start the sensor driver."""
        logger.info(f"Started BRIO {self.device_name}")
        self._is_running = True
        return self._is_running

    def stop(self):
        """Stop the sensor driver."""
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        self._is_running = False
        logger.info(f"Stopped BRIO {self.device_name}")

    def __del__(self):
        """Ensure cleanup when the object is destroyed."""
        if self._is_running:
            self.stop()


def main():
    """Main function to run the BRIO sensor."""
    brio_camera = BRIOSensor(device=0)

    if not brio_camera.is_running:
        print("Failed to start BRIO Sensor.")
        return

    print("Press 'q' to exit the camera stream...")

    while True:
        frame = brio_camera.read()
        if frame is None:
            continue

        frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
        cv2.imshow("BRIO Camera Stream", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    brio_camera.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
