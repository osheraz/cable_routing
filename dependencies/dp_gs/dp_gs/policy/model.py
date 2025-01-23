import torch 
import torch.nn as nn
import torchvision
from typing import Callable
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from dp_gs.policy.utils import ConditionalUnet1D
import torch.nn.functional as F
from dp_gs.util.args import ModelConfig, SharedConfig

def get_resnet(name:str, weights=None, **kwargs) -> nn.Module:
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", None
    """
    # Use standard ResNet implementation from torchvision
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)

    # remove the final fully connected layer
    # for resnet18, the output dim should be 512
    resnet.fc = torch.nn.Identity()
    return resnet


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module

class MLPActionRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(MLPActionRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action = self.fc3(x)
        return action
    
class SimplePolicy(nn.Module):
    hidden_dim = 256
    action_dim = 20
    lowdim_obs_dim = 20
    vision_feature_dim = 512
    def __init__(
        self, 
        model_cfg : ModelConfig,
        shared_cfg : SharedConfig, 
    ):
        super(SimplePolicy, self).__init__()
        self.vision_encoder = get_resnet('resnet18')
        self.vision_encoder = replace_bn_with_gn(self.vision_encoder)
        self.num_cameras = shared_cfg.num_cameras
        # create network object
        self.obs_horizon = shared_cfg.seq_length
        self.action_horizon = shared_cfg.num_pred_steps
        self.only_vision = model_cfg.policy_cfg.only_vision
        if self.only_vision:
            obs_dim = self.vision_feature_dim * self.num_cameras
        else:
            obs_dim = self.vision_feature_dim * self.num_cameras + self.lowdim_obs_dim
        self.global_cond_dim = obs_dim * self.obs_horizon
        
        self.pred_left_only = model_cfg.policy_cfg.pred_left_only
        if self.pred_left_only:
            self.action_dim = self.action_dim // 2
            print("pred_left_only is enabled, changed action dim to ", self.action_dim)
        
        self.mlp = MLPActionRegressor(
            self.global_cond_dim, 
            self.hidden_dim, 
            self.action_dim * self.action_horizon
        )

    def forward(self, nbatch):
        nimage = nbatch["observation"][:, :, :self.num_cameras] # pick the first image # B, T, 1, C, H ,W
        nagent_pos = nbatch["proprio"] # pick the current proprio # B, D
        naction = nbatch['action'][:, -1, ..., :self.action_dim] # B, N, D # enforce pred left only if action dim is changed
        B = nagent_pos.shape[0]

        # encoder vision features
        image_features = self.nets['vision_encoder'](nimage.flatten(end_dim=2)) # to incorporate num_camera dimension
        # convert first to (B, obs_horizon, self.num_cameras, D) then flatten to (B, obs_horizon, self.num_cameras * D)
        image_features = image_features.reshape(*nimage.shape[:3],-1).view(*nimage.shape[:2],-1)

        # concatenate vision feature and low-dim obs
        if not self.only_vision:
            obs_features = torch.cat([image_features, nagent_pos], dim=-1)
        else:
            obs_features = image_features
        obs_cond = obs_features.flatten(start_dim=1)
        # naction = naction.flatten(start_dim=1)
        # (B, obs_horizon * obs_dim)

        # Predict action using MLP
        pred_action = self.mlp(obs_cond)
        pred_action = pred_action.view(B, self.action_horizon, self.action_dim)

        # calculate loss 
        loss = nn.functional.l1_loss(pred_action, naction)
        return loss

class DiffusionPolicy(nn.Module):
    action_dim = 20
    lowdim_obs_dim = 20
    vision_feature_dim = 512
    def __init__(
        self, 
        model_cfg : ModelConfig,
        shared_cfg : SharedConfig, 
    ): 
        super().__init__()
        self.obs_horizon = shared_cfg.seq_length
        self.action_horizon = shared_cfg.num_pred_steps
        self.pred_left_only = model_cfg.policy_cfg.pred_left_only
        self.num_cameras = shared_cfg.num_cameras
        # construct ResNet18 encoder
        # if you have multiple camera views, use seperate encoder weights for each view.
        vision_encoder = get_resnet('resnet18')

        # IMPORTANT!
        # replace all BatchNorm with GroupNorm to work with EMA
        # performance will tank if you forget to do this!
        self.vision_encoder = replace_bn_with_gn(vision_encoder)

        # observation feature has 514 dims in total per step
        self.only_vision = model_cfg.policy_cfg.only_vision
        if self.only_vision:
            obs_dim = self.vision_feature_dim * self.num_cameras
        else:
            obs_dim = self.vision_feature_dim * self.num_cameras + self.lowdim_obs_dim

        # create network object
        self.global_cond_dim = obs_dim * self.obs_horizon

        if self.pred_left_only:
            self.action_dim = self.action_dim // 2
            print("pred_left_only is enabled, changed action dim to ", self.action_dim)

        self.down_dims = model_cfg.policy_cfg.down_dims
        print("UNet latent dimensions: ", self.down_dims)
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            down_dims=self.down_dims,
            global_cond_dim=self.global_cond_dim
        )

        # the final arch has 2 parts
        self.nets = nn.ModuleDict({
            'vision_encoder': self.vision_encoder,
            'noise_pred_net': self.noise_pred_net
        })

        # for this demo, we use DDPMScheduler with 100 diffusion iterations
        self.num_diffusion_iters = model_cfg.policy_cfg.num_train_diffusion_steps
        self.prediction_type='epsilon'
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule='squaredcos_cap_v2',
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type=self.prediction_type
        )

    def forward_loss(self, nbatch):
        # TODO: update when get to it 
        nimage = nbatch["observation"][:, :, :self.num_cameras] # pick the first image # B, T, self.num_cameras, C, H ,W
        nagent_pos = nbatch["proprio"] # pick the current proprio # B, D
        naction = nbatch['action'][:, -1, ..., :self.action_dim] # B, N, D # dataset returns action for all T steps. We only need the last one
        B = nagent_pos.shape[0]

        # encoder vision features
        image_features = self.nets['vision_encoder'](nimage.flatten(end_dim=2)) # to incorporate num_camera dimension
        # convert first to (B, obs_horizon, self.num_cameras, D) then flatten to (B, obs_horizon, self.num_cameras * D)
        image_features = image_features.reshape(*nimage.shape[:3],-1).view(*nimage.shape[:2],-1)

        # concatenate vision feature and low-dim obs
        if not self.only_vision:
            obs_features = torch.cat([image_features, nagent_pos], dim=-1)
        else:
            obs_features = image_features
        obs_cond = obs_features.flatten(start_dim=1)
        # naction = naction.flatten(start_dim=1)
        # (B, obs_horizon * obs_dim)

        # sample noise to add to actions
        noise = torch.randn(naction.shape, device=naction.device)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=naction.device
        ).long()

        # add noise to the clean images according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_actions = self.noise_scheduler.add_noise(
            naction, noise, timesteps)

        # predict the noise residual
        noise_pred = self.noise_pred_net(noisy_actions, timesteps, global_cond=obs_cond)

        # L2 loss
        loss = nn.functional.mse_loss(noise_pred, noise)
        return loss
        
    def forward(self, nbatch):
        # if self.training:
        return self.forward_loss(nbatch)
        # else:
            # raise NotImplementedError("Inference not implemented yet")