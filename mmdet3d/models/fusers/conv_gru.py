from typing import List, Optional

import torch
import torch.nn as nn

from mmdet3d.models.builder import FUSERS

__all__ = ["ConvGRUFuser"]


@FUSERS.register_module()
class ConvGRUFuser(nn.Module):
    """Fuser with a ConvGRU hidden layer for temporal BEV fusion.

    Sensor features are concatenated and projected to `out_channels`, forming
    the GRU input x_t.  The hidden state h_{t-1} is stored as a buffer and
    updated in-place each forward pass.  Call reset_hidden_state() between
    scenes so stale context from the previous sequence is not carried over.

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
        self.input_proj = nn.Sequential(
            nn.Conv2d(sum(in_channels), out_channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # GRU gates operate on (x_t, h_{t-1}) concatenated → out_channels.
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

        # Hidden state — shape is set on the first forward call.
        self.register_buffer("hidden_state", torch.zeros(1), persistent=False)
        self._hidden_initialised = False

    # ------------------------------------------------------------------
    def reset_hidden_state(self) -> None:
        """Zero the hidden state.  Call this at the start of each new scene."""
        self._hidden_initialised = False

    def _init_hidden(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    # ------------------------------------------------------------------
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        # Fuse sensor streams into the GRU input x_t.
        x = self.input_proj(torch.cat(inputs, dim=1))

        # Lazily initialise / reset the hidden state when shape changes.
        if not self._hidden_initialised or self.hidden_state.shape != x.shape:
            self.hidden_state = self._init_hidden(x)
            self._hidden_initialised = True

        h = self.hidden_state

        xh = torch.cat([x, h], dim=1)
        z = self.gate_update(xh)           # update gate
        r = self.gate_reset(xh)            # reset gate
        xrh = torch.cat([x, r * h], dim=1)
        h_candidate = self.candidate(xrh)  # candidate hidden state
        h_new = (1 - z) * h + z * h_candidate

        # Detach so gradients don't propagate across scene boundaries.
        self.hidden_state = h_new.detach()

        return h_new
