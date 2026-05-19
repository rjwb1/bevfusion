"""Merge BEVFusion det + seg pretrained checkpoints into one for det+seg training.

Takes the lidar encoder, fuser, decoder, and object head from the detection
checkpoint, the map head from the segmentation checkpoint, and (by default)
the camera branch from the detection checkpoint.

Note on camera transfer: the released bevfusion-det.pth uses ResNet-50, so the
backbone weights load cleanly into a ResNet-50 config. The neck/vtransform
were trained with different channel widths than the local
configs/nuscenes/det+seg/resnet50-* configs, so those layers will trigger
shape-mismatch warnings at training time and stay at their init_cfg / random
init. The backbone alone is enough to keep fp16 stable through warmup.

Use --no-camera to skip the camera overlay entirely (recovers the old
behavior). Use --camera SRC to pull camera weights from a third checkpoint.
"""
import argparse
from collections import defaultdict

import torch


CAMERA_PREFIX = "encoders.camera."
MAP_HEAD_PREFIX = "heads.map."
DECODER_PREFIX = "decoder."


def state(ckpt):
    return ckpt.get("state_dict", ckpt)


def top_prefix(key):
    parts = key.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else key


def summarize(sd, label):
    counts = defaultdict(int)
    for k in sd:
        counts[top_prefix(k)] += 1
    print(f"{label}: {len(sd)} tensors")
    for p in sorted(counts):
        print(f"  {p}: {counts[p]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--det", default="pretrained/bevfusion-det.pth")
    parser.add_argument("--seg", default="pretrained/bevfusion-seg.pth")
    parser.add_argument("--camera", default=None,
                        help="Source for encoders.camera.* keys. Defaults to --det.")
    parser.add_argument("--no-camera", action="store_true",
                        help="Skip the camera branch entirely (rely on init_weights at train time).")
    parser.add_argument("--multi-decoder", nargs="+", default=["object", "map"],
                        help="Task names to fan decoder.* keys into (e.g. object map). "
                             "Pass --multi-decoder '' to keep a single shared decoder.")
    parser.add_argument("--out", default="pretrained/bevfusion-det+seg-merged.pth")
    args = parser.parse_args()

    # Normalize: an empty string or [""] means "single decoder, no fan-out".
    multi_decoder_tasks = [t for t in args.multi_decoder if t]

    det_sd = state(torch.load(args.det, map_location="cpu"))
    seg_sd = state(torch.load(args.seg, map_location="cpu"))

    camera_src = None
    camera_sd = None
    if not args.no_camera:
        camera_src = args.camera or args.det
        camera_sd = det_sd if camera_src == args.det else state(torch.load(camera_src, map_location="cpu"))

    merged = {}

    # Base: everything from det except camera (will be re-added below), map head,
    # and the decoder (handled separately so we can fan it out if requested).
    for k, v in det_sd.items():
        if (
            k.startswith(CAMERA_PREFIX)
            or k.startswith(MAP_HEAD_PREFIX)
            or k.startswith(DECODER_PREFIX)
        ):
            continue
        merged[k] = v

    # Decoder: either copied straight through, or fanned out per task.
    decoder_keys = [k for k in det_sd if k.startswith(DECODER_PREFIX)]
    if multi_decoder_tasks:
        for task in multi_decoder_tasks:
            for k in decoder_keys:
                # decoder.backbone.X -> decoder.<task>.backbone.X
                new_key = f"{DECODER_PREFIX}{task}.{k[len(DECODER_PREFIX):]}"
                merged[new_key] = det_sd[k]
    else:
        for k in decoder_keys:
            merged[k] = det_sd[k]

    # Map head from seg.
    map_keys = [k for k in seg_sd if k.startswith(MAP_HEAD_PREFIX)]
    for k in map_keys:
        merged[k] = seg_sd[k]

    # Camera from chosen source.
    camera_keys = []
    if camera_sd is not None:
        camera_keys = [k for k in camera_sd if k.startswith(CAMERA_PREFIX)]
        for k in camera_keys:
            merged[k] = camera_sd[k]

    print(f"det: {args.det} ({len(det_sd)} tensors)")
    print(f"seg: {args.seg} ({len(seg_sd)} tensors)")
    if camera_src:
        print(f"camera: {camera_src} ({len(camera_keys)} camera keys)")
    else:
        print("camera: SKIPPED (--no-camera)")
    if multi_decoder_tasks:
        print(
            f"decoder: fanned {len(decoder_keys)} keys -> "
            f"{len(multi_decoder_tasks) * len(decoder_keys)} ({', '.join(multi_decoder_tasks)})"
        )
    else:
        print(f"decoder: shared ({len(decoder_keys)} keys)")
    print()
    summarize(merged, f"merged -> {args.out}")

    if not camera_keys:
        print("\nWARNING: no encoders.camera.* keys in output. The camera branch will be")
        print("         initialized from scratch at train time (backbone via init_cfg,")
        print("         neck/vtransform random). Use a gentle warmup + dynamic loss_scale")
        print("         to avoid fp16 NaN.")

    torch.save(
        {
            "state_dict": merged,
            "meta": {
                "source_det": args.det,
                "source_seg": args.seg,
                "source_camera": camera_src,
                "note": (
                    "Backbone keys transfer to any standard ResNet-50 config. Neck/vtransform "
                    "channel widths depend on the local config and may shape-mismatch at load "
                    "time (skipped by strict=False)."
                ),
            },
        },
        args.out,
    )


if __name__ == "__main__":
    main()
