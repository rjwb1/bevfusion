from typing import List, Optional

import torch
import torch.nn as nn

from mmdet3d.models.builder import FUSERS

__all__ = ["ConvGRUFuser"]


@FUSERS.register_module()
class ConvGRUFuser(nn.Module):
    """Fuser with a ConvGRU recurrence for temporal BEV fusion.

    Clip-based / BPTT semantics
    ---------------------------
    This fuser treats the **batch dimension as time**.  One forward call
    receives the features of a single clip of ``T`` consecutive frames
    (frame ``t`` at batch index ``t``, in temporal order) and runs a ConvGRU
    over them, carrying the hidden state from frame to frame.  The graph is
    kept across the clip, so the recurrence is trained with full
    backpropagation-through-time.  The hidden state is initialised to zeros at
    the start of every clip, so there is no state leakage between clips and no
    persistent buffer to manage.

    Sensor features are concatenated and projected to ``out_channels`` to form
    the GRU input x_t (the ``input_proj`` block mirrors :class:`ConvFuser`, so
    pretrained ConvFuser weights can be loaded straight into it).

    The batch (= clip length) must contain frames of a single sequence in
    order; the :class:`ClipBatchSampler` / :class:`TemporalNuScenesDataset`
    pair guarantees this.  At ``T == 1`` (e.g. plain single-frame inference)
    the fuser degrades gracefully to a single GRU step from a zero state.

    Args:
        in_channels: Per-sensor channel counts (same as ConvFuser).
        out_channels: Hidden-state / output channel count.
        kernel_size: Spatial kernel for all convolutions. Default: 3.
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        pad = kernel_size // 2

        # Project concatenated sensor features to out_channels (the GRU input).
        # Layout matches ConvFuser (Conv2d, BatchNorm2d, ReLU) so a pretrained
        # ConvFuser checkpoint can be remapped onto this block.
        self.input_proj = nn.Sequential(
            nn.Conv2d(sum(in_channels), out_channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # GRU gates operate on (x_t, h_{t-1}) concatenated -> out_channels.
        gru_in = out_channels * 2
        self.gate_update = nn.Sequential(
            nn.Conv2d(gru_in, out_channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid(),
        )
        self.gate_reset = nn.Sequential(
            nn.Conv2d(gru_in, out_channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid(),
        )
        self.candidate = nn.Sequential(
            nn.Conv2d(gru_in, out_channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Tanh(),
        )

    # ------------------------------------------------------------------
    def _step(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """One ConvGRU step. x, h: (N, C, H, W) with N the spatial batch."""
        xh = torch.cat([x, h], dim=1)
        z = self.gate_update(xh)            # update gate
        r = self.gate_reset(xh)             # reset gate
        xrh = torch.cat([x, r * h], dim=1)
        h_candidate = self.candidate(xrh)   # candidate hidden state
        return (1 - z) * h + z * h_candidate

    # ------------------------------------------------------------------
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        # Fuse sensor streams into the per-frame GRU input x_t.
        # Each tensor is (T, C_i, H, W) with T = clip length along the batch dim.
        x = self.input_proj(torch.cat(inputs, dim=1))  # (T, C, H, W)

        T = x.shape[0]
        h = torch.zeros_like(x[:1])  # (1, C, H, W) zero hidden state for frame 0

        outputs = []
        for t in range(T):
            h = self._step(x[t : t + 1], h)  # keep graph -> BPTT across the clip
            outputs.append(h)

        return torch.cat(outputs, dim=0)  # (T, C, H, W)
