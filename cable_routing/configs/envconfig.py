from typing import Literal, Optional, Tuple, Union
import pathlib
import tyro
import dataclasses
from autolab_core import RigidTransform, Point
import numpy as np
import os


@dataclasses.dataclass(frozen=True)
class EnvConfig:
    board_width: float = 0.5
    board_height: float = 0.5
    board_thickness: float = 0.01
    board_offset: float = 0.1
    board_rotation: float = 0.0
    board_translation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    above_clip_1: RigidTransform = RigidTransform(
        rotation=[[-1, 0, 0], [0, 1, 0], [0, 0, -1]], translation=[0.3, -0.2, 0.15]
    )
    # TODO ..


@dataclasses.dataclass(frozen=True)
class YuMiConfig:
    YUMI_MIN_POS: Tuple[float, ...] = (-2.94, -2.00, -2.94, -2.16, -5.06, -1.54, -4.00)
    LEFT_HOME_POS: Tuple[float, ...] = (
        -0.9512928592851002,
        -0.826257179789771,
        1.1051303596535273,
        0.3525379799595703,
        1.1488412220991615,
        0.7785103632911491,
        0.9975875206203292,
    )

    RIGHT_HOME_POS: Tuple[float, ...] = (
        0.814048250667439,
        -0.9499458324057382,
        -1.162067545336931,
        0.3735491891168322,
        -1.3118667613864166,
        0.7759426099471219,
        -1.5252212515862071,
    )


@dataclasses.dataclass(frozen=True)
class BrioConfig:
    width: int = 3840
    height: int = 2160
    width_: int = 3809
    height_: int = 2121
    frame: str = "brio"
    fx: float = 3.43246678e03
    fy: float = 3.44478930e03
    cx: float = 1.79637288e03
    cy: float = 1.08661527e03
    FPS: int = 30
    BRIO_DIST: Tuple[float, ...] = (
        0.22325137,
        -0.73638229,
        -0.00355125,
        -0.0042986,
        0.96319653,
    )

    def get_intrinsic_matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )


@dataclasses.dataclass(frozen=True)
class ZedMiniConfig:
    fps: int = 30
    id: int = 22008760
    resolution: str = "1080p"
    flip_mode: bool = False
    gain: int = 31
    exposure: int = 90
    width: int = 1920
    height: int = 1080
    frame: str = "zed"
    fx: float = 1508.93408203125
    fy: float = 1508.93408203125
    cx: float = 963.8297729492188
    cy: float = 554.3792114257812
    FPS: int = 30

    def get_intrinsic_matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )


@dataclasses.dataclass
class ExperimentConfig:

    camera_cfg: BrioConfig
    robot_cfg: YuMiConfig
    board_cfg: EnvConfig

    cam_to_robot_trans_path: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "cameras/zed_to_world.tf"
    )
    board_cfg_path: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "board/board_config.json"
    )
    bg_img_path: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "board/board_setup.png"
    )
    save_folder: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../trace_test"
    )


if __name__ == "__main__":
    args = tyro.cli(ExperimentConfig)
    dict_args = dataclasses.asdict(args)
    print(dict_args)
