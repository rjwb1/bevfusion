"""DETR-style set-prediction head for line/centreline detection in BEV.

Consumes the per-task decoder feature (the same [B, C, H, W] BEV tensor the map
head uses) and predicts a fixed set of ``num_queries`` line instances, each as a
confidence plus ``num_points`` ordered points in normalised BEV coords ([0, 1]
over the xbound/ybound extent). Training uses Hungarian matching against the
variable number of GT lines; unmatched queries are pushed to "no line".

The intended use is crop-row centrelines, but nothing here is row-specific - it
detects arbitrary BEV polylines. This head is additive: it is only built when a
config declares a ``line`` head, so existing object/map models are unaffected.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import HEADS

__all__ = ["BEVLineHead"]


def _sine_pos_embed(d_model: int, h: int, w: int, device, temperature: float = 10000.0):
    """Standard 2D sine positional embedding, returns [h*w, d_model]."""
    assert d_model % 4 == 0, "d_model must be divisible by 4 for 2D sine embedding"
    num_feats = d_model // 2
    y = torch.arange(h, device=device, dtype=torch.float32)
    x = torch.arange(w, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")  # [h, w]
    dim_t = torch.arange(num_feats // 2, device=device, dtype=torch.float32)
    dim_t = temperature ** (2 * dim_t / (num_feats // 2))
    pos_x = xx.flatten()[:, None] / dim_t  # [hw, num_feats//2]
    pos_y = yy.flatten()[:, None] / dim_t
    pos_x = torch.stack([pos_x.sin(), pos_x.cos()], dim=2).flatten(1)
    pos_y = torch.stack([pos_y.sin(), pos_y.cos()], dim=2).flatten(1)
    return torch.cat([pos_y, pos_x], dim=1)  # [hw, d_model]


class _MLP(nn.Module):
    def __init__(self, dim, hidden, out, layers=3):
        super().__init__()
        dims = [dim] + [hidden] * (layers - 1) + [out]
        self.layers = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(layers))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


@HEADS.register_module()
class BEVLineHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        num_queries: int = 16,
        num_points: int = 2,
        d_model: int = 128,
        nhead: int = 8,
        num_decoder_layers: int = 3,
        ffn_channels: int = 256,
        dropout: float = 0.1,
        cost_point: float = 5.0,
        cost_conf: float = 1.0,
        loss_point_weight: float = 5.0,
        loss_conf_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.xbound = xbound
        self.ybound = ybound
        self.num_queries = num_queries
        self.num_points = max(2, int(num_points))
        self.d_model = d_model
        self.cost_point = cost_point
        self.cost_conf = cost_conf
        self.loss_point_weight = loss_point_weight
        self.loss_conf_weight = loss_conf_weight

        self.input_proj = nn.Conv2d(in_channels, d_model, kernel_size=1)
        self.query_embed = nn.Embedding(num_queries, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward=ffn_channels, dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.conf_head = nn.Linear(d_model, 1)
        self.point_head = _MLP(d_model, d_model, self.num_points * 2, layers=3)

    def _forward_features(self, x: torch.Tensor):
        """x: [B, C, H, W] -> logits [B, N, 1], points [B, N, num_points, 2] in [0,1]."""
        if isinstance(x, (list, tuple)):
            x = x[0]
        B, _, H, W = x.shape
        feat = self.input_proj(x)                      # [B, d, H, W]
        memory = feat.flatten(2).transpose(1, 2)       # [B, HW, d]
        pos = _sine_pos_embed(self.d_model, H, W, x.device).unsqueeze(0)  # [1, HW, d]
        memory = memory + pos
        tgt = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)      # [B, N, d]
        hs = self.decoder(tgt, memory)                 # [B, N, d]
        logits = self.conf_head(hs)                    # [B, N, 1]
        points = self.point_head(hs).sigmoid()         # [B, N, num_points*2] in [0,1]
        points = points.view(B, self.num_queries, self.num_points, 2)
        return logits, points

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[List[torch.Tensor]] = None,
    ) -> Union[Dict[str, torch.Tensor], Dict[str, Any]]:
        logits, points = self._forward_features(x)
        if self.training:
            return self.loss(logits, points, targets)
        return {"line_scores": logits.sigmoid().squeeze(-1), "line_points": points}

    @torch.no_grad()
    def _match(self, pred_pts: torch.Tensor, pred_logit: torch.Tensor, tgt: torch.Tensor):
        """Hungarian match for one sample. Returns (row_ind, col_ind, use_reverse[N,M])."""
        N = pred_pts.shape[0]
        M = tgt.shape[0]
        pf = pred_pts.reshape(N, 1, -1)                       # [N,1,2K]
        tf = tgt.reshape(1, M, -1)                            # [1,M,2K]
        tr = torch.flip(tgt, dims=[1]).reshape(1, M, -1)      # reversed point order
        cost_fwd = (pf - tf).abs().mean(-1)
        cost_rev = (pf - tr).abs().mean(-1)
        cost_pt, rev_idx = torch.min(torch.stack([cost_fwd, cost_rev], 0), dim=0)  # [N,M]
        cost = self.cost_point * cost_pt - self.cost_conf * pred_logit.sigmoid().unsqueeze(1)
        row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
        return row_ind, col_ind, rev_idx

    def loss(self, logits, points, targets):
        B = logits.shape[0]
        device = logits.device
        point_loss = points.sum() * 0.0   # keep grad even with zero matches
        conf_loss = logits.sum() * 0.0
        num_matched = 0
        for b in range(B):
            pred_logit = logits[b].squeeze(-1)             # [N]
            pred_pts = points[b]                           # [N, K, 2]
            target_conf = torch.zeros(self.num_queries, device=device)
            tgt = targets[b] if targets is not None else None
            if tgt is not None and tgt.numel() > 0 and tgt.shape[0] > 0:
                tgt = tgt.to(device).float()
                row_ind, col_ind, rev_idx = self._match(pred_pts, pred_logit, tgt)
                for r, c in zip(row_ind, col_ind):
                    t = tgt[c]
                    if rev_idx[r, c] == 1:
                        t = torch.flip(t, dims=[0])
                    point_loss = point_loss + (pred_pts[r] - t).abs().mean()
                    target_conf[r] = 1.0
                    num_matched += 1
            conf_loss = conf_loss + F.binary_cross_entropy_with_logits(pred_logit, target_conf)
        point_loss = point_loss / max(1, num_matched)
        conf_loss = conf_loss / B
        return {
            "line/point": self.loss_point_weight * point_loss,
            "line/conf": self.loss_conf_weight * conf_loss,
        }
