from autolab_core import CameraIntrinsics, PointCloud, RgbCloud
import pyzed.sl as sl

class Zed:
    def __init__(self, flip_mode, resolution, fps, cam_id=None, recording_file=None, start_time=0.0):
        """ Initializes the ZED camera with the given parameters. """
        self.cam = sl.Camera()
        self.init_params = sl.InitParameters()
        self.init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Enable depth processing
        self.init_params.coordinate_units = sl.UNIT.MILLIMETER
        self.init_params.camera_fps = fps
        self.init_params.sdk_verbose = 1
        self.recording_file = recording_file
        self.start_time = start_time

        if cam_id is not None:
            self.init_params.set_from_serial_number(cam_id)
        if recording_file is not None:
            self.init_params.set_from_svo_file(recording_file)

        self.init_params.camera_image_flip = sl.FLIP_MODE.ON if flip_mode else sl.FLIP_MODE.OFF

        # Set resolution
        if resolution == '720p':
            self.init_params.camera_resolution = sl.RESOLUTION.HD720
        elif resolution == '1080p':
            self.init_params.camera_resolution = sl.RESOLUTION.HD1080
        elif resolution == '2k':
            self.init_params.camera_resolution = sl.RESOLUTION.HD2K
        else:
            raise ValueError("Only 720p, 1080p, and 2k resolutions are supported by ZED.")

        if self.cam.open(self.init_params) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("Failed to open ZED camera.")

        self.runtime_params = sl.RuntimeParameters()

        # Camera Intrinsics & Transformations
        calib = self.cam.get_camera_information().camera_configuration.calibration_parameters
        self.f_ = calib.left_cam.fx
        self.cx_ = calib.left_cam.cx
        self.cy_ = calib.left_cam.cy
        self.Tx_ = calib.stereo_transform.m[0, 3] / 1000  # Convert to meters
        self.cx_diff = calib.right_cam.cx - calib.left_cam.cx

    def get_rgb_depth(self):
        """ Captures left, right images and depth from the ZED camera. """
        if self.cam.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
            left_image, depth = sl.Mat(), sl.Mat()
            # Retrieve depth map. Depth is aligned on the left image
            self.cam.retrieve_image(left_image, sl.VIEW.LEFT)
            # self.cam.retrieve_image(right_image, sl.VIEW.RIGHT)
            self.cam.retrieve_measure(depth, sl.MEASURE.DEPTH)
            return left_image.get_data(), depth.get_data() #, right_image.get_data()
        return None, None

    def get_point_cloud(self):
        """ Retrieves the 3D point cloud from the ZED camera. """
        if self.cam.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
            point_cloud = sl.Mat()
            self.cam.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            return point_cloud.get_data()
        return None

    def start_record(self, out_path):
        """ Starts recording video from the ZED camera. """
        recording_parameters = sl.RecordingParameters(out_path, sl.SVO_COMPRESSION_MODE.H264)
        self.cam.enable_recording(recording_parameters)

    def stop_record(self):
        """ Stops the ZED camera recording. """
        self.cam.disable_recording()

    def get_intrinsics(self):
        """ Returns the camera intrinsic parameters. """
        calib = self.cam.get_camera_information().camera_configuration.calibration_parameters
        return CameraIntrinsics(
            frame="zed",
            fx=calib.left_cam.fx,
            fy=calib.left_cam.fy,
            cx=calib.left_cam.cx,
            cy=calib.left_cam.cy,
            width=self.cam.get_camera_information().camera_resolution.width,
            height=self.cam.get_camera_information().camera_resolution.height,
        )

    def get_stereo_transform(self):
        """ Retrieves the stereo transformation matrix. left-to-right transfromation. """
        transform = self.cam.get_camera_information().camera_configuration.calibration_parameters.stereo_transform.m
        transform[:3, 3] /= 1000  # Convert to meters
        return transform

    def close(self):
        """ Closes the ZED camera connection. """
        self.cam.close()
