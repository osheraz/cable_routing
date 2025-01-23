#  python script/train.py --dataset-cfg.dataset-json config/dataset_config_simple.json --logging-cfg.output-dir /shared/projects/icrl/dp_outputs --logging-cfg.log-name 241118_1151 --trainer-cfg.epochs 100 --shared-cfg.num-pred-steps 16 --shared-cfg.seq-length 12 --dataset-cfg.proprio-noise 0.005 --dataset-cfg.action-noise 0 --dataset-cfg.non-overlapping 32 --shared-cfg.split-epoch 1 --shared-cfg.save-every 10 --dataset-cfg.rebalance-tasks --dataset-cfg.shuffle-repeat-traj --optimizer-cfg.weight-decay 1e-6 --optimizer-cfg.lr 1e-4
import torch
from timm.data.loader import MultiEpochsDataLoader
from dp_gs.policy.model import DiffusionPolicy
from dp_gs.util.args import InferenceConfig
from dp_gs.dataset.dataset import SequenceDataset
import tyro
from dp_gs.dataset.utils import default_vision_transform as transforms_noaug_train
from pathlib import Path
import yaml
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import matplotlib.pyplot as plt
import os
import json
from dp_gs.dataset.utils import unscale_action
import numpy as np
from dp_gs.policy.diffusion_wrapper import DiffusionWrapper
from dp_gs.dataset.image_dataset import SequenceDataset, collate_fn_lambda, VideoSampler
from dp_gs.dataset.image_dataset_sim import SequenceDataset as SimSequenceDataset
class CollateFunction:
    def __init__(self, sequence_length):
        self.sequence_length = sequence_length
        
    def __call__(self, batch):
        return collate_fn_lambda(batch, sequence_length=self.sequence_length)
    
if __name__ == '__main__':
    # parsing args 
    args = tyro.cli(InferenceConfig)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    vision_transform = transforms_noaug_train() # default img_size: Union[int, Tuple[int, int]] = 224,

    train_yaml_path = os.path.join(args.model_ckpt_folder, 'run.yaml')
    train_args = yaml.load(Path(train_yaml_path).read_text(), Loader=yaml.Loader)

    train_args.dataset_cfg.action_statistics = os.path.join(args.model_ckpt_folder, 'action_statistics.json')
    
    dataset_train = SimSequenceDataset(
        dataset_config=train_args.dataset_cfg,
        shared_config=train_args.shared_cfg,
        logging_config=train_args.logging_cfg,
        vision_transform=vision_transform,
        split="val",
        debug=False,
    )

    sampler = VideoSampler(
        dataset_train, 
        batch_size=train_args.shared_cfg.batch_size, 
        sequence_length=train_args.shared_cfg.seq_length
    )

    collate_fn = CollateFunction(train_args.shared_cfg.seq_length)
    dataloader = MultiEpochsDataLoader(
        dataset_train,
        batch_sampler=sampler,
        num_workers=train_args.trainer_cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=4,
    )

    inferencer = DiffusionWrapper(args.model_ckpt_folder, args.ckpt_id, device=device)
   
    for idx, nbatch in enumerate(dataloader):
        # data normalized in dataset
        # device transfer
        nbatch = {k: v.to(device) for k, v in nbatch.items()}
        pred_action = inferencer(nbatch, True)
        proprio = nbatch['proprio'].detach().to('cpu').numpy()
        T = proprio.shape[1]
        gt_action = nbatch['action'][:,-1]#.detach().to('cpu').numpy()
        gt_action = unscale_action(gt_action, stat=inferencer.stats, type='diffusion').detach().to('cpu').numpy()
        #subplot of 20
        fig, axes = plt.subplots(5, 4)
        for i in range(20):
            ax = axes[i//4,i%4]
            ax.plot(range(T), proprio[0, :, i], label='proprio', color='green')
            ax.plot(range(T, T+pred_action[0, :, i].shape[0]),pred_action[0, :, i], label='pred', color='red')
            ax.plot(range(T, T+gt_action[0, :, i].shape[0]),gt_action[0, :, i], label='gt', color='blue')
            
        # plt.plot(pred_action[0, :, 0], label='pred', color='red')
        # plt.plot(gt_action[0, :, 0], label='gt', color='blue')
        # plt.ylim(-1, 1)
        plt.legend()
        fig.savefig(f'pred_vs_gt_{idx}.png')
        if idx > 10:
            break