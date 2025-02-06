
import logging
import numpy as np
import cv2
import matplotlib.pyplot as plt
from autolab_core import CameraIntrinsics

logger = logging.getLogger(__name__)

class SensorIOException(Exception):
    pass

class BRIOSensor():
    """A driver for BRIO .

    Parameters
    ----------
    device : int
        v4l2 port (/dev/video<#> number) for the BRIO camera to connect to.
    """

    def __init__(self, device, inpaint=False, video=True, crop=(0,0,0,0)):
        self._device = int(device)
        self._is_running = False

        # Set up camera intrinsics for the sensor
        # these are the default, but user should check the image size and 
        # appropriately set them if need be
        self.width = 3840 # 3804 after undistortion
        self.height = 2160 # 2119 after undistortion
        self._camera_intr = self.create_intr(self.width, self.height)
        self.BRIO_DIST = np.asarray([ 0.22325137, -0.73638229, -0.00355125, -0.0042986,   0.96319653])
        self.newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(self._camera_intr.K, self.BRIO_DIST, (self.width,self.height), 1, (self.width,self.height))
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(self._camera_intr.K, self.BRIO_DIST, None, self.newcameramtx, (self.width,self.height), 5)

        self._crop = crop
        self._inpaint = inpaint
        self.video = video
        self.initialize()
        
    def initialize(self):
        from os import walk
        device_id_found = False
        # print("Available devices:")
        f = []
        for (dirpath, dirnames, filenames) in walk('/dev'):
            for filename in filenames:
                if filename.startswith('video'):
                    # print(filename)
                    if str(self._device) in filename:
                        device_id_found = True
        if not device_id_found:
            raise SensorIOException(f"Device {self._device} not found; check ls /dev/video*")
                        
        self.cap = cv2.VideoCapture(self._device)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) # turn the autofocus off
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        if not self.video:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if np.any(self._crop):
            self._camera_intr = self._camera_intr.crop(*self._crop)
        
        ret, frame = self.cap.read()
        self._is_running = ret
        if not ret:
            print("Failed to initialize BRIO camera, retry")
            self.cap.release()
        # import pdb; pdb.set_trace()

    @staticmethod
    def create_intr(width,height):
        return CameraIntrinsics(fx=3.43246678e+03,fy=3.44478930e+03,
                cx=1.79637288e+03,cy=1.08661527e+03,width=width,height=height,frame='brio')
        
    @property
    def device_name(self):
        """int : The number of the PhoXi camera to connect to.
        """
        return self._device

    @property
    def is_running(self):
        """bool : True if the stream is running, or false otherwise.
        """
        return self._is_running

    @property
    def intrinsics(self):
        """ the camera intrinsics for this PhoXi camera"""
        return self._camera_intr
    
    @intrinsics.setter
    def intrinsics(self,new_in):
        self._camera_intr=new_in
    @property
    def ir_intrinsics(self):
        return self._camera_intr
    @property
    def ir_frame(self):
        return self.intrinsics.frame

    @property 
    def frame(self):
        return self.intrinsics.frame

    def frames(self):
        #need this to work with CameraChessboardRegistration
        im = self.read()
        return im

    def read(self):
        """Read data from the sensor and return it.
        Returns
        -------
        data : :class:`.Image`
            The rgb image from the sensor.
        """
        # print("Reading from BRIO")
        self._color_im = None

        if not self.video:
            ret, frame = self.cap.read()
            
        ret, frame = self.cap.read()
        self._is_running = ret
        if not ret:
            print("Re-initializing BRIO camera...")
            self.cap.release()
            self._is_running = False
            cv2.destroyAllWindows()
            self.initialize()
            if not self.video:
                ret, frame = self.cap.read()
                
            ret, frame = self.cap.read()

        dst = cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR)
        # crop the image
        x, y, w, h = self.roi
        color = dst[y:y+h, x:x+w] # (2119, 3804, 3)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        
        return color

    def start(self):
        """Start the sensor driver.
        """
        logger.info('Started BRIO {}'.format(self.device_name))
        self._is_running = True
        return self._is_running

    def stop(self):
        """Stop the sensor driver.
        """
        self.cap.release()
        cv2.destroyAllWindows()
        self._is_running = False
        logger.info('Stopped BRIO {}'.format(self.device_name))


    def __del__(self):
        """Automatically stop the sensor for safety.
        """
        if self._is_running:
            self.stop()


import cv2
import time

def main():
    # Initialize the BRIO Sensor with device 0 (default webcam)
    brio_camera = BRIOSensor(device=0)

    # Check if the camera is running
    if not brio_camera.is_running:
        print("Failed to start BRIO Sensor.")
        return

    print("Press 'q' to exit the camera stream...")

    while True:
        frame = brio_camera.read()

        # cv2.imshow("BRIO Camera Stream", frame)
        frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
        cv2.imshow("BRIO Camera Stream", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup: Stop the camera and close the OpenCV window
    brio_camera.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
