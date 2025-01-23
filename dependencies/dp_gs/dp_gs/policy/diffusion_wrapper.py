import torch
from timm.data.loader import MultiEpochsDataLoader
from dp_gs.policy.model import DiffusionPolicy, SimplePolicy
from dp_gs.util.args import InferenceConfig
from dp_gs.dataset.dataset import SequenceDataset
import tyro
# from timm.data.transforms_factory import transforms_noaug_train
from dp_gs.dataset.utils import default_vision_transform as transforms_noaug_train # scale to 224 224 first 
from pathlib import Path
import yaml
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import matplotlib.pyplot as plt
import os
import json
from dp_gs.dataset.utils import unscale_action
import numpy as np
from PIL import Image
from dp_gs.util.args import ExperimentConfig
from dp_gs.policy.undo_vision_tf import undo_vision_transform

def load_state_dict_flexible(model, state_dict):
    """
    Load state dict while handling both DDP and non-DDP scenarios.
    """
    # Check if state_dict has 'module.' prefix
    is_parallel = any(key.startswith('module.') for key in state_dict.keys())
    
    # If model is DDP but state_dict is not parallel
    if hasattr(model, 'module') and not is_parallel:
        model.module.load_state_dict(state_dict)
    
    # If model is not DDP but state_dict is parallel
    elif not hasattr(model, 'module') and is_parallel:
        prefix = 'module.'
        new_state_dict = {key.removeprefix(prefix): value 
                         for key, value in state_dict.items()}
        model.load_state_dict(new_state_dict)
    
    # If both are parallel or both are not parallel
    else:
        model.load_state_dict(state_dict)
        
    return model

class DiffusionWrapper():
    def __init__(self, model_ckpt_folder, ckpt_id, device, denoising_step=100) -> None:
        train_yaml_path = os.path.join(model_ckpt_folder, 'run.yaml')
        model_ckpt_name = os.path.join(model_ckpt_folder, f'checkpoint_{ckpt_id}.pt')
        
        action_stats = json.load(open(os.path.join(model_ckpt_folder, 'action_statistics.json')))
        action_shape = action_stats["shape"]
        min_action = np.array(action_stats["min_action"]).reshape(action_shape)
        max_action = np.array(action_stats["max_action"]).reshape(action_shape)
        self.stats = {
            "min" : torch.from_numpy(min_action),
            "max" : torch.from_numpy(max_action),
        }
        
        args : ExperimentConfig = yaml.load(Path(train_yaml_path).read_text(), Loader=yaml.Loader)
        self.device = device
        if args.model_cfg.policy_type == "diffusion": 
            policy = DiffusionPolicy
            self.policy_type = "diffusion"
        elif args.model_cfg.policy_type == "simple":
            policy = SimplePolicy
            self.policy_type = "simple"
        self.model = policy(
            model_cfg=args.model_cfg,
            shared_cfg=args.shared_cfg,
        ).to(device)
        
        self.model = load_state_dict_flexible(self.model, torch.load(model_ckpt_name))
        self.model.eval()
        
        self.vision_transform = transforms_noaug_train()
        if self.policy_type == "diffusion":
            # self.inference_step = 10
            # self.noise_scheduler = DDIMScheduler(
            #     num_train_timesteps = self.model.num_diffusion_iters,
            #     beta_schedule='squaredcos_cap_v2',
            #     clip_sample=True,
            #     prediction_type= self.model.prediction_type)
            

            self.inference_step = self.model.num_diffusion_iters
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps = self.model.num_diffusion_iters,
                beta_schedule='squaredcos_cap_v2',
                clip_sample=True,
                prediction_type= self.model.prediction_type)
            self.noise_scheduler.set_timesteps(self.inference_step)
    
    def forward_diffusion(self, nbatch, denormalize=True):
        with torch.inference_mode():    
            nimage = nbatch["observation"][:, :self.model.obs_horizon, :self.model.num_cameras].to(self.device) 
            B, T = nimage.shape[:2]
            nimage = nimage.reshape(B*T*self.model.num_cameras, *nimage.shape[3:])
            if nimage.shape[1] != 3:
                nimage = nimage.permute(0, 3, 1, 2) # T, C, H, W
            nimage = self.vision_transform(nimage).float()

            # # for debugging: 
            # # save images and undo vision transform
            # nimage_org = undo_vision_transform(nimage)
            # Image.fromarray(nimage_org[0]).save('nimage_org.png')

            # import pdb; pdb.set_trace()
            nimage = nimage.reshape(B, T, self.model.num_cameras, *nimage.shape[1:])
            
            nagent_pos = nbatch["proprio"][:, :self.model.obs_horizon].to(self.device) # pick the current proprio
            batch_size = nagent_pos.shape[0]
            
            # encoder vision features
            image_features = self.model.nets['vision_encoder']( nimage.flatten(end_dim=2))
            image_features = image_features.reshape(*nimage.shape[:3],-1).view(*nimage.shape[:2],-1)

            # concatenate vision feature and low-dim obs
            if not self.model.only_vision:
                obs_features = torch.cat([image_features, nagent_pos], dim=-1)
            else:
                obs_features = image_features
            obs_cond = obs_features.flatten(start_dim=1)

            naction = torch.randn( (batch_size, self.model.action_horizon, self.model.action_dim), device=self.device)
            
            self.noise_scheduler.set_timesteps(self.inference_step)
            
            for _, k in enumerate(self.noise_scheduler.timesteps):
                # predict noise
                # naction = self.noise_scheduler.scale_model_input(naction, k)
                
                noise_pred = self.model.noise_pred_net( naction,k, obs_cond)
                # inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample
            
            if denormalize:
                if self.model.pred_left_only:
                    naction = torch.concatenate([naction, naction], dim=-1)
                naction = unscale_action(naction, stat=self.stats, type='diffusion')
            
            # handle pred left only 
            if self.model.pred_left_only:
                proprio_right = nbatch["proprio"][:, -1:, self.model.action_dim:]
                naction[:, :, self.model.action_dim:] = proprio_right

            naction = naction.detach().to('cpu').numpy()
            # action_pred = unnormalize_data(naction, stats=stats['action']) todo: implement unnormalize_data
            return naction # [B, T, D]

    def forward_simple(self, nbatch, denormalize=True):
        with torch.inference_mode():
            nimage = nbatch["observation"][:, :self.model.obs_horizon, :self.model.num_cameras].to(self.device) 
            B, T = nimage.shape[:2]
            nimage = nimage.reshape(B*T*self.model.num_cameras, *nimage.shape[3:])
            if nimage.shape[1] != 3:
                nimage = nimage.permute(0, 3, 1 , 2) # T, C, H, W
            nimage = self.vision_transform(nimage).float()
            # import pdb; pdb.set_trace()
            nimage = nimage.reshape(B, T, self.model.num_cameras, *nimage.shape[1:])

            
            nagent_pos = nbatch["proprio"][:, :self.model.obs_horizon].to(self.device) # pick the current proprio
            batch_size = nagent_pos.shape[0]
            
            # encoder vision features
            image_features = self.model.vision_encoder(nimage.flatten(end_dim=2))
            image_features = image_features.reshape(*nimage.shape[:3],-1).view(*nimage.shape[:2],-1)

            # concatenate vision feature and low-dim obs
            if not self.model.only_vision:
                obs_features = torch.cat([image_features, nagent_pos], dim=-1)
            else:
                obs_features = image_features
            obs_cond = obs_features.flatten(start_dim=1)
            # naction = naction.flatten(start_dim=1)
            # (B, obs_horizon * obs_dim)

            # Predict action using MLP
            pred_action = self.model.mlp(obs_cond)
            naction = pred_action.view(B, self.model.action_horizon, self.model.action_dim)
            if denormalize:
                if self.model.pred_left_only:
                    naction = torch.concatenate([naction, naction], dim=-1)
                naction = unscale_action(naction, stat=self.stats, type='diffusion')
            
            # handle pred left only 
            if self.model.pred_left_only:
                proprio_right = nbatch["proprio"][:, -1:, self.model.action_dim:]
                naction[:, :, self.model.action_dim:] = proprio_right

            naction = naction.detach().to('cpu').numpy()
            return naction

    def __call__(self, nbatch, denormalize=True):
        if self.policy_type == "diffusion":
            return self.forward_diffusion(nbatch, denormalize)
        elif self.policy_type == "simple":
            return self.forward_simple(nbatch, denormalize)
