# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import os
import time
import cv2

import torch
import torch.distributed as dist
from tqdm import tqdm

from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.utils.comm import get_world_size, synchronize
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.engine.inference import inference

from maskrcnn_benchmark.structures.image_list import to_image_list

from apex import amp
import numpy as np
from torch.autograd import Variable


from thop import profile

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    cfg,
    model,
    source_data_loader,
    target_data_loader,
    da_enable,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(source_data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()

    data_iter = iter(source_data_loader)
    tgt_data_iter = iter(target_data_loader)


    for iteration in range(max_iter):
        images, images_inpaint, targets = next(data_iter)
        t_images, t_images_inpaint, t_targets = next(tgt_data_iter)
        epoch_period = cfg.SOLVER.EPOCH_PERIOD
        if any(len(target) < 1 for target in targets):
            logger.error(f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}" )
            continue
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        s_images = images.to(device)
        s_images_inpaint = Variable(images_inpaint[0]).to(device)
        s_targets = [target.to(device) for target in targets]

        if da_enable:
            epoch_cur = int((iteration) // epoch_period) + 1
            max_epoch = max_iter / epoch_period
            train_ratio = (float(epoch_cur) / max_epoch)
            grl_alpha = (2. / (1. + np.exp(-10 * train_ratio)) - 1) * 0.01
            t_images = t_images.to(device)
            t_images_inpaint = Variable(t_images_inpaint[0]).to(device)
            t_targets = [target.to(device) for target in t_targets]
        else:
            t_images = None
            t_images_inpaint = None
            t_targets = None
            grl_alpha = None

        
        loss_dict, inpaint_image, filled_image, gt_image, t_inpaint_image, t_filled_image, t_gt_image = model(s_images, s_images_inpaint, t_images, t_images_inpaint, s_targets,t_targets, grl_alpha)
        
        #if loss_dict == None:
        #    continue

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        with amp.scale_loss(losses, optimizer) as scaled_losses:
            scaled_losses.backward()
        optimizer.step()
        scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        
        image_save_interval = 1000
        if iteration % image_save_interval == 0:
            image_save_dir = os.path.join(checkpointer.save_dir, 'images')
            os.makedirs(image_save_dir, exist_ok=True)
            concat_image = torch.cat((inpaint_image, filled_image, gt_image), 3)
            save_image = ((concat_image[0].detach().cpu().numpy().swapaxes(0, 1).swapaxes(1, 2) + 1)*(255/2)).astype(np.uint8)
            cv2.imwrite(os.path.join(image_save_dir, str(int(iteration / image_save_interval)) + '_s.png'), save_image)
            t_concat_image = torch.cat((t_inpaint_image, t_filled_image, t_gt_image), 3)
            t_save_image = ((t_concat_image[0].detach().cpu().numpy().swapaxes(0, 1).swapaxes(1, 2) + 1)*(255/2)).astype(np.uint8)
            cv2.imwrite(os.path.join(image_save_dir, str(int(iteration / image_save_interval)) + '_t.png'), t_save_image)

        if (iteration + 1) % epoch_period == 0 and iteration > 0:
            epoch = int((iteration + 1) / epoch_period)
            checkpointer.save("model_epoch_{:03d}".format(epoch), **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
