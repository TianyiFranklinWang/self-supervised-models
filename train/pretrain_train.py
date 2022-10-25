import datetime
import math
import os
from functools import partial

import torch
import wandb
from functorch.compile import memory_efficient_fusion
from torch.nn.parallel import DistributedDataParallel
from torchvision import datasets

import timm.utils
from model import create_model
from timm.models import model_parameters
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from timm.utils import NativeScaler, reduce_tensor
from util.helper import count_parameters, get_lr, resume_checkpoint, save_model, seed_everything
from util.logger import PretrainMeter, save_files_to_wandb


def train_one_epoch(
        config,
        epoch,
        epochs,
        model,
        loader,
        device,
        optimizer,
        amp_autocast,
        loss_scaler,
        lr_scheduler,
        log_folder,
        meter
):
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    verbose = (rank == 0)

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    if verbose:
        meter.init_metrics(epoch, epochs)
        meter.start_timer('current_epoch_time')

    model.train()
    num_batches_per_epoch = len(loader)
    num_updates = epoch * num_batches_per_epoch
    for batch_idx, (samples, _) in enumerate(loader):
        samples = samples.to(device, non_blocking=True)

        with amp_autocast():
            loss = model(samples)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise RuntimeError(f"Loss is {loss_value}, training protocol aborted")

        optimizer.zero_grad()
        loss_scaler(
            loss,
            optimizer,
            clip_grad=config.clip_grad,
            clip_mode=config.clip_mode,
            parameters=model_parameters(model, exclude_head='agc' in config.clip_mode),
            create_graph=second_order
        )
        if hasattr(model.module, 'update_moving_average'):
            model.module.update_moving_average()

        torch.cuda.synchronize()

        reduced_loss = reduce_tensor(loss.data, world_size)
        if verbose:
            meter.metrics['avg_loss'].update(reduced_loss.item(), samples.size(0))

        num_updates += 1
        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates)

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    torch.distributed.barrier()
    if verbose:
        meter.stop_timer('current_epoch_time')

        lr = get_lr(optimizer)
        meter.update_metrics(lr, 'current_epoch_time')

        print(f"    - ", end='')
        meter.print_metrics()

        meter.update_history()

        if not config.debug:
            meter.plot_dashboard(x='epoch', y='loss', title='Training Loss')
            meter.plot_dashboard(x='epoch', y='lr', title='Learning Rate')

        if config.use_wandb and (not config.debug):
            wandb.log({
                "epoch": meter.metrics['epoch'],
                "lr": meter.metrics['lr'],
                "train/loss": meter.metrics['avg_loss']
            })

        if meter.metrics['avg_loss'] < meter.best_loss:
            meter.best_loss = meter.metrics['avg_loss']
            if config.save_best and (not config.debug):
                file_name = f"{config.model_name}_best.pt"
                save_model(config, meter.metrics['epoch'], model, loss_scaler, optimizer, log_folder, file_name)
                print("\t Best model saved", end='')

        if config.save_interval > 0 and (meter.metrics['epoch'] % config.save_interval == 0):
            file_name = f"{config.model_name}_epoch{meter.metrics['epoch']}.pt"
            save_model(config, meter.metrics['epoch'], model, loss_scaler, optimizer, log_folder, file_name)
            print("\t Intermediate model saved", end='')
        print("")


def pretrain_train_main(config, device, log_folder=None):
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    verbose = (rank == 0)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    if verbose:
        print(f"    - Seeding everything with seed {config.global_seed}")
    seed_everything(seed=config.global_seed, rank=rank)

    if config.jit_fuser is not None and config.aot_autograd:
        if verbose:
            print(f"    - Setting jit fuser type to {config.jit_fuser}")
        timm.utils.set_jit_fuser(config.jit_fuser)

    if verbose:
        print(f'    - Creating model {config.model_name}')
    model = create_model(
        model_name=config.model_name,
        image_size=config.image_size,
        **config.model_kwargs
    ).to(device=device)
    model.zero_grad()
    if verbose:
        print(f"        - Image size: {config.image_size}")
        print(f"        - Custom model setting: {config.model_kwargs}")
        print(f"        - Total params: {count_parameters(model, all=True)}")
        print(f"        - Trainable params: {count_parameters(model, all=False)}")

    if config.aot_autograd:
        if verbose:
            print(f"        - Performing memory efficient fusion")
        model = memory_efficient_fusion(model)

    if config.lr is None:
        if verbose:
            print(f"    - Calculating learning rate")
        global_batch_size = config.batch_size * world_size
        batch_ratio = global_batch_size / config.lr_base_size
        config.lr = config.lr_base * batch_ratio
        if verbose:
            print(f"        - Learning rate: {config.lr}")
            print(f"        - Base learning rate: {config.lr_base}")
            print(f"        - Local batch size: {config.batch_size}")
            print(f"        - Global batch_size: {global_batch_size}")
    else:
        if verbose:
            print(f"    - Setting learning rate")
            print(f"        - Learning rate: {config.lr}")
            print(f"        - Local batch size: {config.batch_size}")

    if verbose:
        print(f"    - Creating optimizer")
        print(f"        - Optimizer: {config.optimizer}")
        print(f"        - Learning rate: {config.lr}")
        print(f"        - Weight decay: {config.weight_decay}")
        print(f"        - Layer decay: {config.layer_decay}")
        print(f"        - Custom optimizer setting: {config.optimizer_kwargs}")
    optimizer = create_optimizer_v2(model,
                                    opt=config.optimizer,
                                    lr=config.lr,
                                    weight_decay=config.weight_decay,
                                    momentum=config.momentum,
                                    layer_decay=config.layer_decay,
                                    **config.optimizer_kwargs
                                    )

    if verbose:
        print(f"    - Setting mixed precision training")
    amp_autocast = partial(torch.autocast, device_type=device.type, dtype=config.amp_dtype)
    loss_scaler = NativeScaler()

    resume_epoch = None
    if config.resume:
        if verbose:
            print(f"    - Resuming from checkpoint {config.resume}")
        resume_epoch = resume_checkpoint(
            model,
            config.resume,
            optimizer=None if config.no_resume_opt else optimizer,
            loss_scaler=None if config.no_resume_opt else loss_scaler,
            verbose=True if verbose else False
        )

    if verbose:
        print("    - Initializing DistributedDataParallel training")
    model = DistributedDataParallel(model,
                                    device_ids=[device],
                                    find_unused_parameters=config.ddp_find_unused_params)

    if config.use_wandb and (not config.debug):
        if verbose:
            print(f"    - Enabling w&b tracing on {config.model_name} with frequency of {config.wandb_log_freq}")
            wandb.watch(model, log_freq=config.wandb_log_freq)

    if verbose:
        print(f"    - Creating training dataset")
    dataset_train = datasets.ImageFolder(os.path.join(config.data_path, config.train_folder_name),
                                         transform=config.dataset_transform_train)
    if verbose:
        print(f"        - Path to data root: {os.path.join(config.data_path, config.train_folder_name)}")
        print(f"        - Number of samples: {len(dataset_train)}")
        print(f"        - Transforms: {config.dataset_transform_train}")

    if verbose:
        print(f"    - Initializing data loader with distributed sampler")
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=world_size, rank=rank, shuffle=True
    )
    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
        sampler=sampler_train
    )

    num_epochs = config.epochs
    lr_scheduler = None
    if config.scheduler:
        if verbose:
            print(f"    - Creating '{config.scheduler}' learning rate scheduler")
        updates_per_epoch = len(loader_train)
        lr_scheduler, num_epochs = create_scheduler_v2(
            optimizer,
            sched=config.scheduler,
            num_epochs=config.epochs,
            decay_epochs=config.decay_epochs,
            decay_milestones=config.decay_milestones,
            cooldown_epochs=config.cooldown_epochs,
            patience_epochs=config.patience_epochs,
            decay_rate=config.decay_rate,
            min_lr=config.min_lr,
            warmup_lr=config.warmup_lr,
            warmup_epochs=config.warmup_epochs,
            warmup_prefix=config.warmup_prefix,
            noise=config.lr_noise,
            noise_pct=config.lr_noise_pct,
            noise_std=config.lr_noise_std,
            cycle_mul=config.lr_cycle_mul,
            cycle_decay=config.lr_cycle_decay,
            cycle_limit=config.lr_cycle_limit,
            k_decay=config.lr_k_decay,
            updates_per_epoch=updates_per_epoch,
        )

    start_epoch = 0
    if resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        if config.schedule_on_updates:
            lr_scheduler.step_update(start_epoch * updates_per_epoch)
        else:
            lr_scheduler.step(start_epoch)

    if verbose:
        print("\n -> Executing training protocol")
        print(f"    - Training from epoch {start_epoch + 1} to epoch {num_epochs}\n")

    meter = None
    if verbose:
        meter = PretrainMeter(log_folder=log_folder)
        meter.start_timer('total_training_time')
    for epoch in range(start_epoch, num_epochs):
        loader_train.sampler.set_epoch(epoch)

        train_one_epoch(
            config=config,
            epoch=epoch,
            epochs=num_epochs,
            model=model,
            loader=loader_train,
            device=device,
            amp_autocast=amp_autocast,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            lr_scheduler=lr_scheduler if config.scheduler else None,
            log_folder=log_folder,
            meter=meter,
        )

        if config.scheduler:
            lr_scheduler.step(epoch=epoch + 1)

    torch.distributed.barrier()

    if verbose:
        meter.stop_timer('total_training_time')

        if not config.debug:
            print(f"\n    - Saving training history to {os.path.join(log_folder, 'history.csv')}")
            meter.history.to_csv(os.path.join(log_folder, 'history.csv'), index=False)

            if config.save_last:
                file_name = f"{config.model_name}_last.pt"
                save_model(config, num_epochs, model, loss_scaler, optimizer, log_folder, file_name)
                print("    - Last model saved")

            if config.use_wandb:
                print("    - Saving assets to w&b")
                wandb.summary['best_loss'] = meter.best_loss
                file_names = ['config.json', 'history.csv']
                if config.save_best:
                    file_names.append(f'{config.model_name}_best.pt')
                if config.save_last:
                    file_names.append(f'{config.model_name}_last.pt')
                save_files_to_wandb(log_folder, file_names=file_names)

        print(f"\n -> Training summary")
        print(f"    - Best loss: {meter.best_loss :.3f}")
        print(f"    - Total time: {str(datetime.timedelta(seconds=int(meter.timers['total_training_time'])))}")
        print(f'\n -> Training protocol finished')
