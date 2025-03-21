import math
import sys
from typing import Iterable, Union

import torch
import torch.nn as nn
from . import misc

from dp_gs.util.args import ExperimentConfig
from dp_gs.policy.model import SimplePolicy, DiffusionPolicy

def train_one_epoch(model: Union[SimplePolicy, DiffusionPolicy], data_loader: Iterable, 
                    optimizer: torch.optim.Optimizer, lr_scheduler : torch.optim.lr_scheduler,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None, validate=False,
                    args : ExperimentConfig=None):
    if validate:
        model.eval()
        validation_loss = 0
    else:
        model.train()
        optimizer.zero_grad() # Clear gradients only during training

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    accum_iter = args.trainer_cfg.accum_iter

    # breakpoint()
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    for data_iter_step, dataset_item in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        for k, v in dataset_item.items():
            dataset_item[k] = v.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model(dataset_item)
            loss_dict = {}
        
        loss_value = loss.item()
        loss_value_dict = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_value_dict = {k: v / accum_iter for k, v in loss_value_dict.items()}
        if not validate:
            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
        
        # this is different from standard pytorch behavior
        if not validate:
            lr_scheduler.step()

        if (data_iter_step + 1) % accum_iter == 0 and not validate:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if validate:
            validation_loss += loss_value_reduce
        loss_value_dict_reduce = {k: misc.all_reduce_mean(v) for k, v in loss_value_dict.items()}
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            global_step = data_iter_step + len(data_loader) * epoch
            if not validate:
                log_writer.add_scalar('train_loss', loss_value_reduce, global_step)
                for k, v in loss_value_dict_reduce.items():
                    log_writer.add_scalar('train_{}'.format(k), v, global_step)
                log_writer.add_scalar('lr', lr, global_step)
            else:
                log_writer.add_scalar('val_loss', loss_value_reduce, global_step)
                for k, v in loss_value_dict_reduce.items():
                    log_writer.add_scalar('val_{}'.format(k), v, global_step)
    if log_writer is not None and validate:
        validation_loss = validation_loss / len(data_loader)
        log_writer.add_scalar('val_loss_epoch', validation_loss, epoch)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
