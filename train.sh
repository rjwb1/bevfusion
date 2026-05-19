bash scripts/run_local.sh train \
  /home/rob/bevfusion/configs/nuscenes/det+seg-multi/resnet50-convfuser.yaml \
  --run-dir runs/det-seg-multidecoder \
  load_from /home/rob/bevfusion/pretrained/bevfusion-det+seg-merged.pth \
  fp16.loss_scale=dynamic \
  checkpoint_config.by_epoch=False \
  checkpoint_config.interval=2000 \
  checkpoint_config.max_keep_ckpts=3 \
  evaluation.save_best=NDS \
  evaluation.rule=greater