"""Strip geometry-constant buffers from a pretrained checkpoint.

The camera vtransform stores its BEV-grid constants (dx/bx/nx) and the depth
frustum as non-trainable nn.Parameters, so they end up in the state_dict. dx/bx/
nx are all shape (3,), so loading a full-range (+/-54 m) checkpoint into a model
built for a different range silently OVERWRITES the model's correctly-computed
grid with the checkpoint's, reverting the camera BEV size (e.g. 100 -> 180).

These are derived constants, not learned weights, so we simply drop them and let
the model keep the values it computed from its own config. Shape-mismatched
learned tensors (e.g. depthnet last conv when dbound changes, head bev_pos when
the grid changes) are already skipped by load_from(strict=False); this only
removes the dangerous *same-shape* constants.

Usage:
    python tools/strip_pretrained_geometry.py <in.pth> <out.pth>
"""
import argparse

import torch

# Buffer name suffixes that are config-derived geometry, not learned weights.
GEOMETRY_SUFFIXES = (".dx", ".bx", ".nx", ".frustum")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="input checkpoint")
    parser.add_argument("dst", help="output checkpoint")
    args = parser.parse_args()

    ckpt = torch.load(args.src, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)

    dropped = [k for k in sd if k.endswith(GEOMETRY_SUFFIXES)]
    for k in dropped:
        del sd[k]

    if "state_dict" in ckpt:
        ckpt["state_dict"] = sd
    else:
        ckpt = sd

    torch.save(ckpt, args.dst)
    print(f"Dropped {len(dropped)} geometry buffers:")
    for k in dropped:
        print(f"  - {k}")
    print(f"Wrote {args.dst}")


if __name__ == "__main__":
    main()
