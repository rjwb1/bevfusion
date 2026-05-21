from functools import partial

import torch
from mmcv.parallel import MMDistributedDataParallel, collate
from mmcv.runner import (
    DistSamplerSeedHook,
    EpochBasedRunner,
    GradientCumulativeFp16OptimizerHook,
    Fp16OptimizerHook,
    OptimizerHook,
    build_optimizer,
    build_runner,
    get_dist_info,
)
from torch.utils.data import DataLoader

from mmdet3d.runner import CustomEpochBasedRunner

from mmdet3d.datasets.clip_sampler import ClipBatchSampler
from mmdet3d.utils import get_root_logger
from mmdet.core import DistEvalHook
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmdet.datasets.builder import worker_init_fn


def build_clip_dataloader(dataset, workers_per_gpu, shuffle, dist, seed):
    """DataLoader whose every batch is one temporal clip (batch dim == time).

    Used for :class:`TemporalNuScenesDataset` so that :class:`ConvGRUFuser`
    sees ordered, single-scene clips.  ``samples_per_gpu`` is implicitly the
    clip length, so collate stacks each clip into a (T, ...) batch.
    """
    rank, world_size = get_dist_info()
    if not dist:
        world_size, rank = 1, 0

    batch_sampler = ClipBatchSampler(
        dataset.clips,
        shuffle=shuffle,
        num_replicas=world_size,
        rank=rank,
        seed=seed if seed is not None else 0,
    )

    init_fn = (
        partial(worker_init_fn, num_workers=workers_per_gpu, rank=rank, seed=seed)
        if seed is not None
        else None
    )

    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=workers_per_gpu,
        collate_fn=partial(collate, samples_per_gpu=dataset.clip_length),
        pin_memory=False,
        worker_init_fn=init_fn,
    )


def train_model(
    model,
    dataset,
    cfg,
    distributed=False,
    validate=False,
    timestamp=None,
):
    logger = get_root_logger()

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    def _build_train_loader(ds):
        if hasattr(ds, "clips"):
            logger.info(
                f"Temporal dataset detected: {len(ds.clips)} clips of length "
                f"{ds.clip_length} (batch dim == time)."
            )
            return build_clip_dataloader(
                ds,
                cfg.data.workers_per_gpu,
                shuffle=True,
                dist=distributed,
                seed=cfg.seed,
            )
        return build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            None,
            dist=distributed,
            seed=cfg.seed,
        )

    data_loaders = [_build_train_loader(ds) for ds in dataset]

    # put model on gpus
    find_unused_parameters = cfg.get("find_unused_parameters", False)
    # Sets the `find_unused_parameters` parameter in
    # torch.nn.parallel.DistributedDataParallel
    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False,
        find_unused_parameters=find_unused_parameters,
    )

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.run_dir,
            logger=logger,
            meta={},
        ),
    )
    
    if hasattr(runner, "set_dataset"):
        runner.set_dataset(dataset)

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        if "cumulative_iters" in cfg.optimizer_config:
            optimizer_config = GradientCumulativeFp16OptimizerHook(
                **cfg.optimizer_config, **fp16_cfg, distributed=distributed
            )
        else:
            optimizer_config = Fp16OptimizerHook(
                **cfg.optimizer_config, **fp16_cfg, distributed=distributed
            )
    elif distributed and "type" not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get("momentum_config", None),
    )
    if isinstance(runner, EpochBasedRunner):
        runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop("samples_per_gpu", 1)
        if val_samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        if hasattr(val_dataset, "clips"):
            # Temporal eval: ordered, non-overlapping clips that partition every
            # scene, so results stay in dataset-index order for evaluate().
            val_dataloader = build_clip_dataloader(
                val_dataset,
                workers_per_gpu=cfg.data.workers_per_gpu,
                shuffle=False,
                dist=distributed,
                seed=cfg.seed,
            )
        else:
            val_dataloader = build_dataloader(
                val_dataset,
                samples_per_gpu=val_samples_per_gpu,
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=distributed,
                shuffle=False,
            )
        eval_cfg = cfg.get("evaluation", {})
        eval_cfg["by_epoch"] = cfg.runner["type"] != "IterBasedRunner"
        eval_hook = DistEvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, [("train", 1)])
