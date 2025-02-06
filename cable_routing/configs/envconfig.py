from typing import Literal, Optional, Tuple, Union
import pathlib
import tyro
import dataclasses
from autolab_core import RigidTransform, Point

@dataclasses.dataclass(frozen=True)
class EnvConfig:
    board_width: float = 0.5
    board_height: float = 0.5
    board_thickness: float = 0.01
    board_offset: float = 0.1
    board_rotation: float = 0.0
    board_translation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    above_clip_1: RigidTransform = RigidTransform(
        rotation=[[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
        translation=[0.3, -0.2, 0.15]
    )
    # TODO ..
    

@dataclasses.dataclass(frozen=True)
class YuMiConfig:
    YUMI_MIN_POS: Tuple[float, ...] = (-2.94, -2.00, -2.94, -2.16, -5.06, -1.54, -4.00)


@dataclasses.dataclass(frozen=True)
class BrioConfig:
    width: int = 3840
    height: int = 2160
    frame: str = 'brio'
    fx: float = 3.43246678e+03
    fy: float = 3.44478930e+03
    cx: float = 1.79637288e+03
    cy: float = 1.08661527e+03

@dataclasses.dataclass(frozen=True)
class ZedMiniConfig:
    fps: int = 30
    id: int = 22008760
    resolution: str = '1080p'
    flip_mode: bool = False
    gain: int = 31
    exposure: int = 90
    
@dataclasses.dataclass
class ExperimentConfig:
    camera_cfg: BrioConfig
    robot_cfg: YuMiConfig
    board_cfg: EnvConfig
    
if __name__ == "__main__":
    args = tyro.cli(ExperimentConfig)
    dict_args = dataclasses.asdict(args)
    print(dict_args)
    
    