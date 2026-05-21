"""Build a ConvGRU pretrain checkpoint from a trained ConvFuser model.

The ConvGRU and ConvFuser models share every weight except the fuser:
encoders, decoder(s) and heads have identical parameter names and load
directly.  Only the fusion block differs.  ConvGRUFuser keeps a
ConvFuser-shaped projection (``fuser.input_proj.{0,1}``) as the GRU input
stage, so we remap the pretrained ConvFuser conv/BN onto it; the GRU gates
(``gate_update``/``gate_reset``/``candidate``) stay randomly initialised.

The optimizer state and epoch/iter meta are dropped so the result is a clean
``load_from`` initialisation (fresh LR schedule, not a resume).

Usage:
    python tools/convert_convfuser_to_convgru.py SRC.pth DST.pth
"""
import argparse

import torch

# fuser.0 (Conv2d) -> fuser.input_proj.0 ; fuser.1 (BN) -> fuser.input_proj.1
REMAP_PREFIXES = {
    "fuser.0.": "fuser.input_proj.0.",
    "fuser.1.": "fuser.input_proj.1.",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="trained ConvFuser checkpoint")
    parser.add_argument("dst", help="output ConvGRU init checkpoint")
    args = parser.parse_args()

    ckpt = torch.load(args.src, map_location="cpu")
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    new_sd = {}
    remapped = 0
    for k, v in sd.items():
        nk = k
        for src_prefix, dst_prefix in REMAP_PREFIXES.items():
            if k.startswith(src_prefix):
                nk = dst_prefix + k[len(src_prefix) :]
                remapped += 1
                break
        new_sd[nk] = v

    out = {"state_dict": new_sd, "meta": {}}
    torch.save(out, args.dst)
    print(
        f"Wrote {args.dst}: {len(new_sd)} tensors "
        f"({remapped} fuser keys remapped to fuser.input_proj.*; "
        f"GRU gates left for random init)."
    )


if __name__ == "__main__":
    main()
