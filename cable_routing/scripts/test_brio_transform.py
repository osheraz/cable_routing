from autolab_core import RigidTransform, Point, CameraIntrinsics
import tyro
import numpy as np
from cable_routing.configs.envconfig import ExperimentConfig
from cable_routing.env.ext_camera.brio_camera import BRIOSensor
import cv2
import time

TABLE_HEIGHT = 0.0582

def get_world_coord_from_pixel_coord(pixel_coord, cam_intrinsics, cam_extrinsics):
    '''
    pixel_coord: [x, y] in pixel coordinates
    cam_intrinsics: 3x3 camera intrinsics matrix
    '''
    pixel_coord = np.array(pixel_coord)
    point_3d_cam = np.linalg.inv(cam_intrinsics._K).dot(np.r_[pixel_coord, 1.04 - TABLE_HEIGHT])
    point_3d_world = cam_extrinsics.matrix.dot(np.r_[point_3d_cam, 1.0])
    point_3d_world = point_3d_world[:3]/point_3d_world[3]
    point_3d_world[-1] = TABLE_HEIGHT 
    return point_3d_world

def click_event(event, u, v, flags, param):
    """Handles mouse click events to get pixel coordinates."""
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel_value = param["img"][v, u]
        print(f"Pixel coordinates: (u={u}, v={v}) - Pixel value: {pixel_value}")

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(param["img"], f"({u},{v})", (u, v), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(param["img"], str(pixel_value), (u, v + 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        time.sleep(0.5)
        param["u"] = u
        param["v"] = v


# @hydra.main(version_base=None, config_name="config", config_path=get_main_config_dir())
def main(args: ExperimentConfig):
    print("Initializing ...") 
    
    cam_cfg = args.camera_cfg
    cam = BRIOSensor(device=0)
    T_CAM_BASE = RigidTransform.load("/home/osheraz/cable_routing/data/zed/brio_to_world.tf").as_frames(from_frame="brio", to_frame="base_link")

    print(T_CAM_BASE)
    CAM_INTR = CameraIntrinsics(fx=cam_cfg.fx,
                                fy=cam_cfg.fy,
                                cx=cam_cfg.cx,
                                cy=cam_cfg.cy,
                                width=cam_cfg.width,
                                height=cam_cfg.height,
                                frame=cam_cfg.frame
                                )
    frame = cam.read()

    params = {"img": frame.copy(), "u": None, "v": None}
    cv2.imshow("Image", frame)
    cv2.setMouseCallback("Image", click_event, param=params)

    while params["u"] is None or params["v"] is None:
        if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC key
            break

    pixel_coord = [params["u"], params["v"]]
    world_coord = get_world_coord_from_pixel_coord(pixel_coord, CAM_INTR, T_CAM_BASE)
    print("World coordinate: ", world_coord)

if __name__ == '__main__':

    args = tyro.cli(ExperimentConfig)
    main(args)