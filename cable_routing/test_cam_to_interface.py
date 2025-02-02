from yumi_jacobi.interface import Interface
from autolab_core import RigidTransform, Point
import os
from omegaconf import DictConfig
import hydra
from hydra.utils import to_absolute_path
from cable_routing.configs.config import get_main_config_dir, split_main_config

@hydra.main(version_base=None, config_name="config", config_path=get_main_config_dir())
def main(cfg: DictConfig):
    print("Initializing YuMi...")


if __name__ == '__main__':
    print(get_main_config_dir())
    main()