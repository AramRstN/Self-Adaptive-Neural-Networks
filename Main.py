import albumentations
import timm
import time
import torch_pruning as tp
import torch
import torch.utils.data
import torchvision
from torchvision.models import resnet50
from torch import nn
import datetime
import os, sys
import warnings
import registry
from PIL import ImageFile
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
from functools import partial
from engine.utils.imagenet_utils import presets, transforms, utils, sampler
from unittest.mock import patch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import utils1
import copy

#sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=UserWarning)


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument("--data-path", default="~/train_imagenet/data/imagenet/", type=str, help="dataset path")
    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument("--pretrained", action="store_true")

    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("--epochs", default=90, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)")
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument("--wd", "--weight-decay", default=1e-4, type=float, metavar="W", help="weight decay (default: 1e-4)", dest="weight_decay")
    parser.add_argument("--norm-weight-decay", default=None, type=float, help="weight decay for Normalization layers (default: None, same value as --wd)")
    parser.add_argument("--bias-weight-decay", default=None, type=float, help="weight decay for bias parameters of all layers (default: None, same value as --wd)")
    parser.add_argument("--transformer-embedding-decay", default=None, type=float, help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)")
    parser.add_argument("--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing")
    parser.add_argument("--mixup-alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)")
    parser.add_argument("--cutmix-alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)")
    parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument("--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)")
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
    parser.add_argument("--print-freq", default=100, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--cache-dataset", dest="cache_dataset", help="Cache the datasets for quicker initialization. It also serializes the transforms", action="store_true")
    parser.add_argument("--sync-bn", dest="sync_bn", help="Use sync batch norm", action="store_true")
    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true")
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--ra-magnitude", default=9, type=int, help="magnitude of auto augment policy")
    parser.add_argument("--augmix-severity", default=3, type=int, help="severity of augmix policy")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters")
    parser.add_argument("--model-ema-steps", type=int, default=32, help="the number of iterations that controls how often to update the EMA model (default: 32)")
    parser.add_argument("--model-ema-decay", type=float, default=0.99998, help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)")
    parser.add_argument("--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only.")
    parser.add_argument("--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)")
    parser.add_argument("--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)")
    parser.add_argument("--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)")
    parser.add_argument("--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)")
    parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
    parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
    parser.add_argument("--ra-reps", default=3, type=int, help="number of repetitions for Repeated Augmentation (default: 3)")
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    
    # pruning parameters
    parser.add_argument("--prune", action="store_true")
    parser.add_argument("--method", type=str, default='l1')
    parser.add_argument("--global-pruning", default=False, action="store_true")
    parser.add_argument("--target-flops", type=float, default=2.0, help="GFLOPs of pruned model")
    parser.add_argument("--soft-keeping-ratio", type=float, default=0.0)
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--delta_reg", type=float, default=1e-4)
    parser.add_argument("--max-pruning-ratio", default=1.0, type=float, help="maximum channel pruning ratio")
    parser.add_argument("--sl-epochs", type=int, default=None)
    parser.add_argument("--sl-resume", type=str, default=None)
    parser.add_argument("--sl-lr", default=None, type=float, help="learning rate")
    parser.add_argument("--sl-lr-step-size", default=None, type=int, help="milestones for learning rate decay")
    parser.add_argument("--sl-lr-warmup-epochs", default=None, type=int, help="the number of epochs to warmup (default: 0)")
    return parser

def w_result(*args):

    with open('Testresult.txt', 'a') as file:
        sys.stdout = file

        for arg in args:
            print(arg)
        

def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path

def load_data(traindir, valdir, args):
    # Data loading code
    print("Loading data...")
    resize_size, crop_size = (342, 299) if args.model == 'inception_v3' else (256, 224)

    print("Loading training data...")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading dataset_train from {}".format(cache_path))
        dataset, _ = torch.load(cache_path)
    else:
        auto_augment_policy = getattr(args, "auto_augment", None)
        random_erase_prob = getattr(args, "random_erase", 0.0)
        # dataset = torchvision.datasets.ImageFolder(
        #     traindir,
        #     presets.ClassificationPresetTrain(crop_size=crop_size, auto_augment_policy=auto_augment_policy,
        #                                       random_erase_prob=random_erase_prob))


        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = torchvision.datasets.ImageFolder(
            traindir,
            transform=train_transforms)

        
        if args.cache_dataset:
            print("Saving dataset_train to {}...".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)
    print("Data loading took", time.time() - st)

    print("Loading validation data...")
    cache_path = _get_cache_path(valdir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading dataset_test from {}".format(cache_path))
        dataset_test, _ = torch.load(cache_path)
    else:
        # dataset_test = torchvision.datasets.ImageFolder(
        #     valdir,
        #     presets.ClassificationPresetEval(crop_size=crop_size, resize_size=resize_size))

        val_transforms = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset_test = torchvision.datasets.ImageFolder(
            valdir,
            transform=val_transforms)

        
        if args.cache_dataset:
            print("Saving dataset_test to {}...".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    print("Creating data loaders...")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler



def prune_to_target_flops(pruner, model, target_flops, example_inputs):
    model.eval()
    ori_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    pruned_ops = ori_ops
    pruner.step()
    pruned_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    #while pruned_ops / 1e9 > target_flops:
    #    pruner.step()
    #    if 'vit' in args.model:
    #        model.hidden_dim = model.conv_proj.out_channels
    #    pruned_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
        
    return pruned_ops

def prune_model_to_target_percentage(pruner, model, target_percentage, example_inputs):
    model.eval()
    iterative_steps=100
    original_params = tp.utils.count_params(model)
    current_percentage = 0.0

    while current_percentage < target_percentage:
        # Step the pruner to prune a certain percentage of weights
        pruner.step()

        # Calculate the current pruning percentage
        current_params = tp.utils.count_params(model)
        current_percentage = 100.0 * (original_params - current_params) / original_params

        #print(f"Current pruning percentage: {current_percentage:.2f}%")
        w_result(f"Current pruning percentage: {current_percentage:.2f}%")
        sys.stdout = sys.__stdout__

        
        ###### remove ########
        #history = pruner.pruning_history()
        #return model, history
        ###### remove ########
        
    history = pruner.pruning_history()
    #print(history)
    utils1.save_model(model,f"Pruned_{current_percentage:.2f}", True)
    
    return model, history

def get_pruner(model, example_inputs, args):
    unwrapped_parameters = (
        [model.encoder.pos_embedding, model.class_token] if "vit" in args.model else None
    )
    sparsity_learning = False
    data_dependency = False
    if args.method == "random":
        imp = tp.importance.RandomImportance()
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    elif args.method == "l1":
        imp = tp.importance.MagnitudeImportance(p=1)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    elif args.method == "lamp":
        imp = tp.importance.LAMPImportance(p=2)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    elif args.method == "slim":
        sparsity_learning = True
        imp = tp.importance.BNScaleImportance()
        pruner_entry = partial(tp.pruner.BNScalePruner, reg=args.reg, global_pruning=args.global_pruning)
    elif args.method == "group_norm":
        imp = tp.importance.GroupNormImportance(p=2)
        pruner_entry = partial(tp.pruner.GroupNormPruner, global_pruning=args.global_pruning)
    elif args.method == "group_greg":
        sparsity_learning = True
        imp = tp.importance.GroupNormImportance(p=2)
        pruner_entry = partial(tp.pruner.GrowingRegPruner, reg=args.reg, delta_reg=args.delta_reg, global_pruning=args.global_pruning)
    elif args.method == "group_sl":
        sparsity_learning = True
        imp = tp.importance.GroupNormImportance(p=2)
        pruner_entry = partial(tp.pruner.GroupNormPruner, reg=args.reg, global_pruning=args.global_pruning)
    else:
        raise NotImplementedError
    args.data_dependency = data_dependency
    args.sparsity_learning = sparsity_learning
    ignored_layers = []
    pruning_ratio_dict = {}
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
            ignored_layers.append(m)
    round_to = None
    if 'vit' in args.model:
        round_to = model.encoder.layers[0].num_heads
    
    #pruner = tp.pruner.MagnitudePruner(
    #    model,
    #    example_inputs,
    #    importance=imp,
    #    iterative_steps=1,
    #    ch_sparsity=0.25,
    #    ignored_layers=ignored_layers,
    #    global_pruning=True
    #)
    pruner = pruner_entry(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=100,
        pruning_ratio=1.0,
        pruning_ratio_dict=pruning_ratio_dict,
        max_pruning_ratio=args.max_pruning_ratio,
        ignored_layers=ignored_layers,
        round_to=round_to,
        unwrapped_parameters=unwrapped_parameters,
    )
    return pruner

def train(
    model, 
    epochs, 
    lr, lr_step_size, lr_warmup_epochs, 
    train_sampler, data_loader, data_loader_test, 
    device, args, pruner=None, state_dict_only=True, recover=None):

    model.to(device)
    
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.label_smoothing>0:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    weight_decay = args.weight_decay if pruner is None else 0
    bias_weight_decay = args.bias_weight_decay if pruner is None else 0
    norm_weight_decay = args.norm_weight_decay if pruner is None else 0

    custom_keys_weight_decay = []
    if bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
    parameters = utils.set_weight_decay(
        model,
        weight_decay,
        norm_weight_decay=norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=args.momentum,
            weight_decay=weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=lr, momentum=args.momentum, weight_decay=weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    model_ema = None
    if args.model_ema:
        # Decay adjustment that aims to keep the decay independent from other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and ommit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = args.world_size * args.batch_size * args.model_ema_steps / epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if model_ema:
            evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
        else:
            evaluate(model, criterion, data_loader_test, device=device)
        return

    start_time = time.time()
    best_acc = 0
    prefix = '' if pruner is None else 'regularized_{:e}_'.format(args.reg)
    for epoch in range(args.start_epoch, epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema, scaler, pruner, recover=recover)
        lr_scheduler.step()
        acc = evaluate(model, criterion, data_loader_test, device=device)
        if model_ema:
            acc = evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict() if state_dict_only else model_without_ddp,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                # "args": args,
            }
            if model_ema:
                checkpoint["model_ema"] = model_ema.state_dict()
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            if acc>best_acc:
                best_acc=acc
                utils.save_on_master(checkpoint, os.path.join(args.output_dir, prefix+"best.pth"))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, prefix+"latest.pth"))
        #print("Epoch {}/{}, Current Best Acc = {:.6f}".format(epoch, epochs, best_acc))
        w_result("Epoch {}/{}, Current Best Acc = {:.6f}".format(epoch, epochs, best_acc))
        sys.stdout = sys.__stdout__

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    #print(f"Training time {total_time_str}")

    w_result(f"Training time {total_time_str}")
    sys.stdout = sys.__stdout__

    return model_without_ddp

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema=None, scaler=None, pruner=None, recover=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if pruner:
                scaler.unscale_(optimizer)
                pruner.regularize(model)
            #if recover:
            #    recover(model.module)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if pruner is not None:
                pruner.regularize(model)
            if recover:
                recover(model.module)
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)
            
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
        
    if pruner is not None and isinstance(pruner, tp.pruner.GrowingRegPruner):
        pruner.update_reg()

def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
    # gather the stats from all processes

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    #print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    w_result(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    sys.stdout = sys.__stdout__

    return metric_logger.acc1.global_avg

def get_layer(model, submodule_name):
    submodule_names = submodule_name.split('.')
    submodule = model
    for name in submodule_names:
        submodule = getattr(submodule, name)
    return submodule

def get_all_leaf_layers(module):
    for name, m in module.named_children():
        if type(m) in {nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.Linear}:
            yield name
        else:
            yield from (name+'.'+ subm for subm in get_all_leaf_layers(m))

def set_layer(layer_name, pruned_model, layer):
    name_parts = layer_name.split('.')
    last_layer_name = name_parts[-1]
    layer_place = pruned_model
    for name in name_parts[:-1]:
        layer_place = getattr(layer_place, name)
    setattr(layer_place, last_layer_name, layer)

def get_pruned_indices(layer_name, prune_history):
    for ln, _, pruned_indices in prune_history:
        if layer_name == ln:
            return pruned_indices
    return None

def get_in_pruned(pruned_model, groups, layer_name):
    for i, g in enumerate(groups):
        if g[0][0].target.name.split()[0] == layer_name:
            return i
    return -1

def train_freez_mode(rebuilt_model):

    train(rebuilt_model, args.epochs, 
            lr=args.lr, lr_step_size=args.lr_step_size, lr_warmup_epochs=args.lr_warmup_epochs, 
            train_sampler=train_sampler, data_loader=data_loader, data_loader_test=data_loader_test, 
            device=device, args=args, pruner=None, state_dict_only=(not args.prune))
    
    return rebuilt_model
    
def rebuild_layer(original_layer, pruned_layer):
    new_layer = original_layer
# def rebuild_layer(pruned_model, original_layer, pruned_layer_name, pruned_indices, prev_layer_pruned_indices, groups, effected_group):
    #pruned_layer = get_layer(pruned_model, pruned_layer_name)
    #set_layer(pruned_layer_name, pruned_model, original_layer)
    # 2. recover fine-tuned wieghts from pruned layer weights.
    #new_pruned_layer = get_layer(pruned_model, pruned_layer_name)
    # new_layer_chan_size0 = new_pruned_layer.weight.shape[0]
    # remained_from_pruned0 = [i for i in range(new_layer_chan_size0) if not i in pruned_indices]
    # new_layer_chan_size1 = new_pruned_layer.weight.shape[1]
    # remained_from_pruned1 = [i for i in range(new_layer_chan_size1) if not i in prev_layer_pruned_indices]
    #dim0 = groups[effected_group][0][1]
    #dim1 = groups[effected_group+1][0][1]
    with torch.no_grad():
        #print(f'orig_shape={original_layer.weight.shape}\tprun_shape={pruned_layer.weight.shape}\t{pruned_layer_name}\t{len(dim0)}\t{len(dim1) }')
        #new_pruned_layer.weight[dim0][:, dim1] = pruned_layer.weight
        if pruned_layer.weight.ndim == 1:
            new_layer.weight[:pruned_layer.weight.shape[0]] = pruned_layer.weight
        elif pruned_layer.weight.ndim == 2:
            new_layer.weight[:pruned_layer.weight.shape[0], :pruned_layer.weight.shape[1]] = pruned_layer.weight
        elif pruned_layer.weight.ndim == 3:
            new_layer.weight[:pruned_layer.weight.shape[0], :pruned_layer.weight.shape[1], :pruned_layer.weight.shape[2]] = pruned_layer.weight
        elif pruned_layer.weight.ndim == 4:
            new_layer.weight[:pruned_layer.weight.shape[0], :pruned_layer.weight.shape[1], :pruned_layer.weight.shape[2], :pruned_layer.weight.shape[3]] = pruned_layer.weight
        elif pruned_layer.weight.ndim == 5:
            new_layer.weight[:pruned_layer.weight.shape[0], :pruned_layer.weight.shape[1], :pruned_layer.weight.shape[2], :pruned_layer.weight.shape[3], :pruned_layer.weight.shape[4]] = pruned_layer.weight
        elif pruned_layer.weight.ndim == 6:
            new_layer.weight[:pruned_layer.weight.shape[0], :pruned_layer.weight.shape[1]] = ...
        elif pruned_layer.weight.ndim == 7:
            new_layer.weight[:pruned_layer.weight.shape[0], :pruned_layer.weight.shape[1]] = ...

    

    
    return new_layer

def rebuild_model(original_model, pruned_model, prune_history):

    for param in original_model.parameters():
        param.requires_grad = False

#def rebuild_model(original_model, pruned_model, pruner, prune_history):
    #groups = list(pruner.DG.get_all_groups())
    #prev_layer_pruned_indices = []
    
    for i, layer_name in enumerate(get_all_leaf_layers(original_model)):
        for j in range(len(prune_history)):
            if prune_history[j][0] == layer_name and prune_history[j][2]:
                #rebuild_layer()
                pruned_layer = get_layer(pruned_model, layer_name)

                new_layer = get_layer(original_model, layer_name)

                final_layer = rebuild_layer(new_layer, pruned_layer)

                final_layer.requires_grad = True
                for param in final_layer.parameters():
                    param.requires_grad = True

    #train_freez_mode(original_model)
    #return original_model
    return original_model


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)

    collate_fn = None
    num_classes = len(dataset.classes)
    mixup_transforms = []

    if args.mixup_alpha > 0.0:
        mixup_transforms.append(transforms.RandomMixup(num_classes, p=1.0, alpha=args.mixup_alpha))
    if args.cutmix_alpha > 0.0:
        mixup_transforms.append(transforms.RandomCutmix(num_classes, p=1.0, alpha=args.cutmix_alpha))
    if mixup_transforms:
        mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)
        def collate_fn(batch):
            return mixupcutmix(*default_collate(batch))

    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True
    )


    #print("Creating model")
    w_result("Creating model")
    sys.stdout = sys.__stdout__
    
    model = registry.get_model(num_classes=1000, name=args.model, pretrained=args.pretrained, target_dataset='imagenet') #torchvision.models.__dict__[args.model](pretrained=args.pretrained) #torchvision.models.get_model(args.model, weights=args.weights, num_classes=num_classes)
    model.to(device)
    model.eval()
    #print("="*16)
    w_result("="*16)
    sys.stdout = sys.__stdout__
    
    example_inputs = torch.randn(1, 3, 224, 224).to(device)
    base_ops, base_params = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    # print("Params: {:.4f} M".format(base_params / 1e6))
    # print("ops: {:.4f} G".format(base_ops / 1e9))
    # print("="*16)
    w_result("Params: {:.4f} M".format(base_params / 1e6))
    w_result("ops: {:.4f} G".format(base_ops / 1e9))
    w_result("="*16)
    sys.stdout = sys.__stdout__
    
    if args.prune:
        pruner = get_pruner(model, example_inputs=example_inputs, args=args)

        if args.sparsity_learning:
            if args.sl_resume:
                #print("Loading sparse model from {}...".format(args.sl_resume))
                w_result("Loading sparse model from {}...".format(args.sl_resume))
                sys.stdout = sys.__stdout__

                model.load_state_dict( torch.load(args.sl_resume, map_location='cpu')['model'] )
            else:
                #print("Sparsifying model...")
                w_result("Sparsifying model...")
                sys.stdout = sys.__stdout__

                if args.sl_lr is None: args.sl_lr = args.lr
                if args.sl_lr_step_size is None: args.sl_lr_step_size = args.lr_step_size
                if args.sl_lr_warmup_epochs is None: args.sl_lr_warmup_epochs = args.lr_warmup_epochs
                if args.sl_epochs is None: args.sl_epochs = args.epochs
                train(model, args.sl_epochs, 
                                        lr=args.sl_lr, lr_step_size=args.sl_lr_step_size, lr_warmup_epochs=args.sl_lr_warmup_epochs, 
                                        train_sampler=train_sampler, data_loader=data_loader, data_loader_test=data_loader_test, 
                                        device=device, args=args, pruner=pruner, state_dict_only=True)
        #model = model.to('cpu')
        #print("Pruning model...")
        w_result("Pruning model...")
        sys.stdout = sys.__stdout__


        
        #prune_to_target_flops(pruner, model, args.target_flops, example_inputs)
        #pruned_ops, pruned_size = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
        
        original_model = copy.deepcopy(model)
        
        
        target_pruning_percentage = 10.0
        
        pruned_model, pruning_history = prune_model_to_target_percentage(pruner, model, target_pruning_percentage, example_inputs)
        
        #return pruned_model, pruning_history, original_model, pruner
        
        pruned_ops, pruned_size = tp.utils.count_ops_and_params(pruned_model, example_inputs=example_inputs)
        
        # print("="*16)
        # print("After pruning:")
        # #print(model)
        # print("Params: {:.2f} M => {:.2f} M ({:.2f}%)".format(base_params / 1e6, pruned_size / 1e6, pruned_size / base_params * 100))
        # print("Ops: {:.2f} G => {:.2f} G ({:.2f}%, {:.2f}X )".format(base_ops / 1e9, pruned_ops / 1e9, pruned_ops / base_ops * 100, base_ops / pruned_ops))
        # print("="*16)


        w_result("="*16)
        w_result("After pruning:")
        w_result("Params: {:.2f} M => {:.2f} M ({:.2f}%)".format(base_params / 1e6, pruned_size / 1e6, pruned_size / base_params * 100))
        w_result("Ops: {:.2f} G => {:.2f} G ({:.2f}%, {:.2f}X )".format(base_ops / 1e9, pruned_ops / 1e9, pruned_ops / base_ops * 100, base_ops / pruned_ops))
        w_result("="*16)
        sys.stdout = sys.__stdout__
        

    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)

    #print("Finetuning..." if args.prune else "Training...")
    w_result("Finetuning..." if args.prune else "Training...")
    sys.stdout = sys.__stdout__

    train(pruned_model, args.epochs, 
            lr=args.lr, lr_step_size=args.lr_step_size, lr_warmup_epochs=args.lr_warmup_epochs, 
            train_sampler=train_sampler, data_loader=data_loader, data_loader_test=data_loader_test, 
            device=device, args=args, pruner=None, state_dict_only=(not args.prune))
    

    # # rebuils part

    args.prune = False
    w_result("Rebuilding...")
    sys.stdout = sys.__stdout__

    best_model = torch.load('run/imagenet/mobilenet_v2_gnorm/best.pth')
    list(best_model.keys())
    best_model = best_model['model']


    rebuilt_model = rebuild_model(original_model, pruned_model, pruning_history)

    train(rebuilt_model, args.epochs, 
            lr=args.lr, lr_step_size=args.lr_step_size, lr_warmup_epochs=args.lr_warmup_epochs, 
            train_sampler=train_sampler, data_loader=data_loader, data_loader_test=data_loader_test, 
            device=device, args=args, pruner=None, state_dict_only=(not args.prune))


args = get_args_parser().parse_args()
#pruned_model, pruning_history, original_model, pruner = main(args)
main(args)
