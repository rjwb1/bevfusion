bash scripts/run_local.sh train \
  /home/rob/bevfusion/configs/nuscenes/det+seg/resnet50-convfuser.yaml \
  --run-dir runs/det-seg-finetune \
  resume_from runs/det-seg-finetune/latest.pth  \
  optimizer.lr=2.5e-5 \
  lr_config.warmup_iters=2000 \
  lr_config.warmup_ratio=0.333 \
  max_epochs=6 \
  fp16.loss_scale=dynamic \
  checkpoint_config.by_epoch=False \
  checkpoint_config.interval=2000 \
  checkpoint_config.max_keep_ckpts=3 \
  evaluation.save_best=NDS \
  evaluation.rule=greater