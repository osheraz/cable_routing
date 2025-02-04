from typing import Literal, Optional, Tuple, Union
import pathlib
import tyro
import dataclasses

@dataclasses.dataclass
class YuMiConfig:
    YUMI_MIN_POS: Tuple[float, ...] = (-2.94, -2.00, -2.94, -2.16, -5.06, -1.54, -4.00)


@dataclasses.dataclass
class CameraConfig:
    width: int = 3840
    height: int = 2160
    frame: str = 'brio'
    fx: float = 3.43246678e+03
    fy: float = 3.44478930e+03
    cx: float = 1.79637288e+03
    cy: float = 1.08661527e+03


@dataclasses.dataclass
class ExperimentConfig:
    camera_cfg: CameraConfig
    robot_cfg: YuMiConfig

if __name__ == "__main__":
    args = tyro.cli(ExperimentConfig)
    dict_args = dataclasses.asdict(args)
    print(dict_args)