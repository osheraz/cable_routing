# from yumi_jacobi.interface import Interface
from autolab_core import RigidTransform, Point, CameraIntrinsics
from omegaconf import DictConfig
import hydra
from hydra.utils import to_absolute_path
from cable_routing.configs.utils.hydraconfig import get_main_config_dir, split_main_config
import numpy as np

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


@hydra.main(version_base=None, config_name="config", config_path=get_main_config_dir())
def main(cfg: DictConfig):
    print("Initializing ...")
    
    
    T_CAM_BASE =RigidTransform(rotation=[[1.0000000, 0.0000000,  0.0000000], 
                                         [0.0000000, 1.0000000, 0.0000000], 
                                         [0.0000000, 0.0000000, 1.0000000]],
                                        translation=[0.0, 0.0, 0.0])

    CAM_INTR = CameraIntrinsics(fx=cfg.cameras.camera_matrix.fx,
                                fy=cfg.cameras.camera_matrix.fy,
                                cx=cfg.cameras.camera_matrix.cx,
                                cy=cfg.cameras.camera_matrix.cy,
                                width=cfg.cameras.width,
                                height=cfg.cameras.height,
                                frame=cfg.cameras.frame
                                )

    pixel_coord = [320, 240]
    world_coord = get_world_coord_from_pixel_coord(pixel_coord, CAM_INTR, T_CAM_BASE)
    print("World coordinate: ", world_coord)

if __name__ == '__main__':
    
    main()