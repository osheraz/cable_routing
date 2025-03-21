#  CUDA_VISIBLE_DEVICES=4,5,6,7 --main_process_port=12547 accelerate launch script/train.py --dataset-cfg.dataset-root /home/mfu/dataset/dp_gs/transfer_tiger_241204 --logging-cfg.log-name 241211_1403 --logging-cfg.output-dir /shared/projects/icrl/dp_outputs --shared-cfg.no-use-delta-action --shared-cfg.seq-length 4 --shared-cfg.num-pred-steps 16 --dataset-cfg.subsample-steps 4 --trainer-cfg.epochs 300
# from dp_gs.dataset.dataset import SequenceDataset
# from timm.data.loader import MultiEpochsDataLoader
import numpy as np
import os 
import torch
import torch.backends.cudnn as cudnn
import tyro
import wandb
import yaml
import json
import dp_gs.util.misc as misc

from diffusers.optimization import get_scheduler
from dp_gs.dataset.image_dataset_sim import SequenceDataset as SimSequenceDataset
from dp_gs.dataset.image_dataset import SequenceDataset, VideoSampler, collate_fn_lambda
from dp_gs.dataset.utils import default_vision_transform, aug_vision_transform
from dp_gs.policy.model import DiffusionPolicy, SimplePolicy
from dp_gs.util.args import ExperimentConfig
from dp_gs.util.misc import NativeScalerWithGradNormCount as NativeScaler
from dp_gs.util.engine import train_one_epoch

from pathlib import Path
from torch.utils.data import DataLoader 
from torch.utils.tensorboard import SummaryWriter
from dp_gs.util.misc import MultiEpochsDataLoader

class CollateFunction:
    def __init__(self, sequence_length):
        self.sequence_length = sequence_length
        
    def __call__(self, batch):
        return collate_fn_lambda(batch, sequence_length=self.sequence_length)

def main(args : ExperimentConfig):
    # spawn is needed to initialize vision augmentation within pytorch workers
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass  # start method already set
    
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    # fix the seed for reproducibility
    seed = args.shared_cfg.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    start_epoch = args.shared_cfg.start_epoch
    num_epochs = args.trainer_cfg.epochs

    output_dir = args.logging_cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(args.device)

    # accelerator = Accelerator()
    # device = accelerator.device
    
    if args.model_cfg.policy_type == "diffusion":
        policy = DiffusionPolicy
    else:
        policy = SimplePolicy
    model = policy(
        shared_cfg=args.shared_cfg,
        model_cfg=args.model_cfg,
    ).to(device)
    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    eff_batch_size = args.shared_cfg.batch_size * args.trainer_cfg.accum_iter * misc.get_world_size()
    print("accumulate grad iterations: %d" % args.trainer_cfg.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # vision transform
    base_vision_transform = default_vision_transform() # default img_size: Union[int, Tuple[int, int]] = 224,
    aug_transform = aug_vision_transform() # default img_size: Union[int, Tuple[int, int]] = 224,
    print("vision transform: ", default_vision_transform)
    print("augmented vision transform: ", aug_transform)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    if args.dataset_cfg.is_sim_data:
        dataset_type = SimSequenceDataset
    else:
        dataset_type = SequenceDataset

    # datasets
    dataset_train = dataset_type(
        dataset_config=args.dataset_cfg,
        shared_config=args.shared_cfg,
        logging_config=args.logging_cfg,
        vision_transform=aug_transform,
        split="train",
    )
    dataset_val = dataset_type(
        dataset_config=args.dataset_cfg,
        shared_config=args.shared_cfg,
        vision_transform=base_vision_transform,
        logging_config=args.logging_cfg,
        split="val"
    )

    # samplers
    train_sampler = VideoSampler(
        dataset_train, 
        batch_size=args.shared_cfg.batch_size, 
        sequence_length=args.shared_cfg.seq_length, 
        num_replicas=num_tasks,
        rank=global_rank,
    )
    val_sampler = VideoSampler(
        dataset_val, 
        batch_size=args.shared_cfg.batch_size, 
        sequence_length=args.shared_cfg.seq_length,
        num_replicas=num_tasks,
        rank=global_rank,
    )

    collate_fn = CollateFunction(args.shared_cfg.seq_length)
    dataloader_train = MultiEpochsDataLoader(
        dataset_train,
        batch_sampler=train_sampler,
        num_workers=args.trainer_cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=4,
    )
    dataloader_val = MultiEpochsDataLoader(
        dataset_val,
        batch_sampler=val_sampler,
        num_workers=args.trainer_cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=4,
    )

    # Start a wandb run with `sync_tensorboard=True`
    if global_rank == 0 and args.logging_cfg.log_name is not None:
        wandb.init(entity="project_vit", project="dp_gs", config=args, name=args.logging_cfg.log_name, sync_tensorboard=True)

    # SummaryWrite
    if global_rank == 0 and args.logging_cfg.log_dir is not None:
        os.makedirs(args.logging_cfg.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.logging_cfg.log_dir)
    else:
        log_writer = None

    # Standard ADAM optimizer
    # Note that EMA parametesr are not optimized
    param_groups = misc.add_weight_decay(model_without_ddp, args.optimizer_cfg.weight_decay)
    optimizer = torch.optim.AdamW(
        params=param_groups,
        lr=args.optimizer_cfg.lr)
    loss_scaler = NativeScaler()

    # --resume
    misc.resume_from_ckpt(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader_train) * num_epochs
    )

    for epoch_idx in range(start_epoch, num_epochs):
        train_stats = train_one_epoch(
            model=model, data_loader=dataloader_train, 
            optimizer=optimizer, lr_scheduler=lr_scheduler,
            device=device, epoch=epoch_idx,
            loss_scaler=loss_scaler, log_writer=log_writer,
            validate=False, args=args
        )
        
        if dataloader_val is not None:
            with torch.no_grad():
                val_stats = train_one_epoch(
                    model=model, data_loader=dataloader_val, 
                    optimizer=optimizer, lr_scheduler=lr_scheduler,
                    device=device, epoch=epoch_idx,
                    loss_scaler=loss_scaler, log_writer=log_writer,
                    validate=True, args=args
                )
            print("Validation Epoch {}".format(epoch_idx))
        
        # save checkpoint 
        if misc.is_main_process() and (epoch_idx % args.shared_cfg.save_every == 0 or epoch_idx == num_epochs - 1):
            torch.save(model.state_dict(), os.path.join(output_dir, f'checkpoint_{epoch_idx}.pt'))

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch_idx}
        if dataloader_val is not None:
            log_stats.update({f'val_{k}': v for k, v in val_stats.items()})

        if args.logging_cfg.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.logging_cfg.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

if __name__ == '__main__':
    # parsing args 
    args = tyro.cli(ExperimentConfig)

    if args.load_config is not None: 
        print("loading configs from file: ", args.load_config)
        assert os.path.exists(args.load_config), f"Config file does not exist: {args.load_config}"
        args : ExperimentConfig = yaml.load(Path(args.load_config).read_text(), Loader=yaml.Loader) 

    # creating the output directory and logging directory 
    if args.logging_cfg.log_name is not None: 
        args.logging_cfg.output_dir = os.path.join(args.logging_cfg.output_dir, args.logging_cfg.log_name)
    if args.logging_cfg.log_dir is None:
        args.logging_cfg.log_dir = args.logging_cfg.output_dir
    if args.logging_cfg.output_dir:
        Path(args.logging_cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # dump the args into a yaml file 
    with open(os.path.join(args.logging_cfg.output_dir, "run.yaml"), 'w') as f:
        yaml.dump(args, f)

    main(args)
