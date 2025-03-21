import json 
import line_profiler
import numpy as np 
import os 
import torch 
import torchvision.transforms as transforms
import zarr 

from glob import glob
from PIL import Image
from torch.utils.data import Sampler, Dataset
from tqdm import tqdm 
from typing import List, Iterator, Optional

from dp_gs.util.args import DatasetConfig, SharedConfig, LoggingConfig
from .utils import quat_to_rot_6d, quat_to_euler, euler_to_quat, convert_multi_step_np, convert_delta_action, scale_action

class VideoSampler(Sampler):
    """
    Sampler for the sequence dataset with epoch-based shuffling.
    For each instance, it will be of size batch_size * sequence_length,
    then the collate_fn will reshape it to batch_size, sequence_length, ...
    """
    def __init__(
        self, 
        data: Dataset, 
        batch_size: int, 
        sequence_length: int, 
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 42
    ) -> None:
        self.data = data
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0

        self.num_chunks = len(self.data) // (self.batch_size * self.sequence_length * self.num_replicas)
        self.total_length = self.num_chunks * self.batch_size * self.sequence_length

        self.start_idx = self.rank * self.total_length
        self.end_idx = self.start_idx + self.total_length

    def __len__(self) -> int:
        return self.num_chunks
    
    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler."""
        self.epoch = epoch
    
    def __iter__(self) -> Iterator[int]:
        """
        Generate indices for each epoch with proper shuffling.
        Uses both epoch and seed for reproducible but different shuffling per epoch.
        """
        # Create a generator with seed that combines epoch and base seed
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        # Generate indices for this process
        indices = torch.arange(start=self.start_idx, end=self.end_idx)
        
        # Shuffle indices while maintaining sequence grouping
        num_batches = len(indices) // (self.batch_size * self.sequence_length)
        indices = indices.view(num_batches, self.batch_size * self.sequence_length)
        
        # Shuffle the batches
        perm = torch.randperm(num_batches, generator=g)
        indices = indices[perm].view(-1)
        
        # Yield batches of indices
        for batch in torch.chunk(indices, self.num_chunks):
            yield batch.tolist()

def collate_fn_lambda(batch : List[dict], sequence_length : int):
    """
    Collate function for the sequence dataset
    note that since we have modified how batch is defined via video sampler, we need to collect the data 
    to the correct order: batch_size * sequence_length, ... -> batch_size, sequence_length, ...
    """
    # get the keys 
    keys = batch[0].keys()
    collated = {}
    for key in keys:
        data = [sample[key] for sample in batch]
        data = torch.stack(data)
        data = data.view(-1, sequence_length, *data.shape[1:])
        collated[key] = data
    return collated

class SequenceDataset(Dataset):
    action_key : str = "action/cartesian_pose" # [LEFT ARM] w, x, y, z, -- x, y, z + [RIGHT ARM] w, x, y, z -- x, y, z
    proprio_key : str = "state/cartesian/cartesian_pose" # [LEFT ARM] w, x, y, z, -- x, y, z + [RIGHT ARM] w, x, y, z -- x, y, z
    joint_key : str = "state/joint/joint_angle_rad" # for getting gripper information # (n, 16) dtype('float64')
    traj_start_idx : int = 0
    gripper_width :  int = 0.0226
    scale_left_gripper : float = 1
    scale_right_gripper : float = 1

    def __init__(
        self, 
        dataset_config : DatasetConfig,
        shared_config : SharedConfig,
        logging_config : LoggingConfig,
        vision_transform : transforms.Compose,
        split : str = "train",
        debug : bool = False,
    ):
        self.seq_length = shared_config.seq_length
        self.num_pred_steps = shared_config.num_pred_steps
        self.dataset_root = dataset_config.dataset_root
        self.subsample_steps = dataset_config.subsample_steps
        self.num_cameras = shared_config.num_cameras
        assert os.path.exists(self.dataset_root), f"Dataset root {self.dataset_root} does not exist"
        self.vision_transform = vision_transform
        
        self.use_delta_action = shared_config.use_delta_action
        self.proprio_noise = dataset_config.proprio_noise

        # calculate length of dataset 
        common_path = glob("**/*proprio.zarr", root_dir=self.dataset_root, recursive=True)
        self.common_path = [os.path.join(self.dataset_root, p.replace("_proprio.zarr", "")) for p in common_path]

        self.file2length = {}
        for file in self.common_path:
            self.file2length[file] = self.get_traj_length(file) - 1 # subtract 1 as we need to predict the next action

        # load action statistics
        if dataset_config.action_statistics is not None:
            assert os.path.exists(dataset_config.action_statistics), f"Action statistics file {dataset_config.action_statistics} does not exist"
            with open(dataset_config.action_statistics, 'r') as f:
                action_stats = json.load(f)
            action_shape = action_stats["shape"]
            min_action = np.array(action_stats["min_action"]).reshape(action_shape)
            max_action = np.array(action_stats["max_action"]).reshape(action_shape)
        else:
            output_dir = logging_config.output_dir
            min_action, max_action = self.calculate_dataset_statistics(os.path.join(output_dir, "action_statistics.json"))
        self.stats = {
            "min" : torch.from_numpy(min_action), 
            "max" : torch.from_numpy(max_action),
        }

        # randomly shuffle the file2length dataset
        rng = np.random.default_rng(seed=shared_config.seed)
        keys = list(self.file2length.keys())
        rng.shuffle(keys)

        # train test split
        if split == "train":
            keys = keys[:int(len(keys) * dataset_config.train_split)]
        else:
            keys = keys[int(len(keys) * dataset_config.train_split):]
        self.file2length = {k : self.file2length[k] for k in keys}

        # index to start end 
        self.start_end = []
        self.debug = debug
        for file_path, length in self.file2length.items():
            for i in range(length - (self.subsample_steps * (self.seq_length + self.num_pred_steps) - 1)):
                self.start_end.append((file_path, i, i + self.subsample_steps * (self.seq_length + self.num_pred_steps)))

        rng = np.random.default_rng(seed=shared_config.seed)
        rng.shuffle(self.start_end)

        # modify start end such that it is flattened in sequence length dimension
        new_start_end = []
        for file_path, start, end in self.start_end:
            for i in range(start, end - self.subsample_steps * self.num_pred_steps, self.subsample_steps):
                new_start_end.append((file_path, i, i + self.subsample_steps * self.num_pred_steps))
        self.start_end = new_start_end

        self.total_length = len(self.start_end)

        # vision augmentation
        if dataset_config.vision_aug:
            self.vision_aug = True
            self.contrast_range = [0.8, 1.2]
            self.brightness_range = [-0.1, 0.1]
            print("using numeric brightness and contrast augmentation")
            print("contrast range: ", self.contrast_range)
            print("brightness range: ", self.brightness_range)
        else:
            self.vision_aug = False

        self.enable_scale_action = dataset_config.scale_action

    def calculate_dataset_statistics(
        self, 
        output_path : str = "config/action_statistics.json"
    ):
        # calculate the min and max of delta actions for left and right arm 
        global_min_action, global_max_action = None, None
        good_files = []
        for file in tqdm(self.common_path):
            left_action, right_action = self.helper_load_action(file, self.traj_start_idx, self.get_traj_length(file) - self.subsample_steps)
            left_proprio, right_proprio = self.helper_load_proprio(file, self.traj_start_idx, self.get_traj_length(file) - self.subsample_steps, False)
            left_action = convert_multi_step_np(left_action, self.num_pred_steps)
            right_action = convert_multi_step_np(right_action, self.num_pred_steps)
            delta_left_action = convert_delta_action(left_action, left_proprio)
            delta_right_action = convert_delta_action(right_action, right_proprio)
            good_files.append(file)
            if self.use_delta_action:
                action = np.concatenate([delta_left_action, delta_right_action], axis=-1)
            else:
                action = np.concatenate([left_action, right_action], axis=-1)

            min_action, max_action = action.min((0,1)), action.max((0,1)) # only calculate on the action dim
            if global_min_action is None:
                global_min_action = min_action
                global_max_action = max_action
            else:
                global_min_action = np.stack([global_min_action, min_action], axis=0).min(0)
                global_max_action = np.stack([global_max_action, max_action], axis=0).max(0)    
        self.common_path = good_files
        
        if np.all(global_max_action[..., 9] <= 0.01):
            self.scale_left_gripper = self.gripper_width / global_max_action[..., 9]
            global_max_action[..., 9] = self.gripper_width
        else:
            self.scale_left_gripper = 1
        
        if np.all(global_max_action[..., 19] <= 0.01):
            self.scale_right_gripper = self.gripper_width / global_max_action[..., 19]
            global_max_action[..., 19] = self.gripper_width
        else:
            self.scale_right_gripper = 1
        # save the statistics 
        stats = {
            "shape" : global_min_action.shape,
            "min_action": global_min_action.flatten().tolist(),
            "max_action": global_max_action.flatten().tolist()
        }
        with open(output_path, 'w') as f:
            json.dump(stats, f)
        print("Action statistics saved to ", output_path)
        return global_min_action, global_max_action
        

    def __len__(self):
        return self.total_length

    def get_traj_length(
        self, 
        file_path : str,
    ): 
        action_fp = file_path + "_proprio.zarr" 
        actions = zarr.load(action_fp)
        return len(actions)
    
    @line_profiler.profile
    def __getitem__(self, idx : int):
        file_path, start, end = self.start_end[idx]
        left_action, right_action = self.helper_load_action(file_path, start, end)
        left_proprio, right_proprio = self.helper_load_proprio(file_path, start, start + 1)
        
        proprio = np.concatenate([left_proprio, right_proprio], axis=-1) 
        proprio = torch.from_numpy(proprio)
        # get rid of the time dimension since there's just one step
        proprio = proprio.squeeze(0)

        # get delta actions 
        if self.use_delta_action:
            left_action = convert_delta_action(left_action[None, :], left_proprio)
            right_action = convert_delta_action(right_action[None, :], right_proprio)
        left_action = torch.from_numpy(left_action)
        right_action = torch.from_numpy(right_action)

        # concatenate actions 
        actions = torch.concatenate([left_action, right_action], dim=-1)

        # get rid of the time dimension since there's just one step
        actions = actions.squeeze(0)

        # option for not scaling 
        if self.enable_scale_action:
            actions = scale_action(actions, self.stats, type="diffusion") 

        # remove nans 
        actions = torch.nan_to_num(actions)
            
        # get camera 
        camera = self.helper_load_camera(file_path, start, end)

        return {
            "action" : actions.float(), # num_pred_steps, 20
            "proprio" : proprio.float(), # 20
            "observation" : camera.float(), # num_camera, 3, 224, 224
        }
        
    def helper_load_action(self, file_path : str, start : int, end : int):
        
        # get action path, joint path, and retrieved indices
        indices = np.arange(start + self.subsample_steps, end + self.subsample_steps, self.subsample_steps)
        action_fp = file_path + "_proprio.zarr"
        joint_fp = file_path + "_joint.zarr"

        # get actions
        actions = zarr.load(action_fp)[indices]
        left, right = actions[:, :7], actions[:, 7:]
        
        # get gripper
        joint_data = zarr.load(joint_fp)[indices]
        left_g = joint_data[:, -1][:, None]
        right_g = joint_data[:, -2][:, None]

        left = np.concatenate([
            left[:, 4:],
            quat_to_rot_6d(left[:, :4]), 
            left_g * self.scale_left_gripper
        ], axis=1)
        right = np.concatenate([
            right[:, 4:],
            quat_to_rot_6d(right[:, :4]), 
            right_g * self.scale_right_gripper
        ], axis=1)
        return left, right

    def randomize(self, transform : np.ndarray):
        # randomize the transform (N, 7) -> (N, 7)
        # each transform is wxyz, xyz
        t, rot = transform[:, 4:7], transform[:, :4]
        t += np.random.uniform(-self.proprio_noise, self.proprio_noise, t.shape)
        rot = quat_to_euler(rot)
        rot += np.random.uniform(-self.proprio_noise, self.proprio_noise, rot.shape)
        rot = euler_to_quat(rot)
        rt = np.concatenate([rot, t], axis=1)
        return rt

    def helper_load_proprio(self, file_path : str, start : int, end : int, noisy : bool = True):
        indices = np.arange(start, end, self.subsample_steps)
        proprio_fp = file_path + "_proprio.zarr"
        joint_fp = file_path + "_joint.zarr"
        
        # get proprio data
        proprio = zarr.load(proprio_fp)[indices]
        left, right = proprio[:, :7], proprio[:, 7:]

        # get gripper data
        joint_data = zarr.load(joint_fp)[indices]
        left_g, right_g = joint_data[:, -1][:, None], joint_data[:, -2][:, None]
        
        # add proprio noise 
        if noisy:
            left = self.randomize(left)
            right = self.randomize(right)

        left = np.concatenate([
            left[:, 4:],
            quat_to_rot_6d(left[:, :4]), 
            left_g * self.scale_left_gripper
        ], axis=1)
        right = np.concatenate([
            right[:, 4:],
            quat_to_rot_6d(right[:, :4]), 
            right_g * self.scale_right_gripper
        ], axis=1)
        return left, right

    @line_profiler.profile
    def helper_load_camera(self, file_path : str, start : int, end : int):
        # indices = np.arange(start, end - self.num_pred_steps * self.subsample_steps, self.subsample_steps)
        # import pdb; pdb.set_trace()
        # assert len(indices) == 1, "now we are only retrieving one frame"
        image_left_path = file_path + f"_left/{start:04d}.jpg"
        image_right_path =  file_path + f"_right/{start:04d}.jpg"
        camera_observations = []
        for image_path in [image_left_path, image_right_path]:
            image = Image.open(image_path)
            image = self.vision_transform(image)
            camera_observations.append(image)

        subsequence = torch.stack(camera_observations) # num_camera, 3, H, W
        return subsequence
