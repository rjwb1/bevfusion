import math
from typing import Iterator, List, Optional

import numpy as np
from torch.utils.data import Sampler

__all__ = ["ClipBatchSampler"]


class ClipBatchSampler(Sampler):
    """Yield one temporal clip per batch.

    Each clip is a list of dataset indices belonging to a single scene, given
    in temporal order.  Used as the ``batch_sampler`` of a ``DataLoader`` so
    that every produced batch is exactly one ordered clip (batch dim == time),
    which is what :class:`ConvGRUFuser` consumes.

    Args:
        clips: list of clips, each a list of dataset indices in temporal order.
        shuffle: shuffle clip *order* between epochs (frames within a clip stay
            ordered).  Use ``True`` for training, ``False`` for evaluation.
        num_replicas / rank: DDP sharding.  Clips are partitioned across ranks.
        seed: base RNG seed for the per-epoch shuffle.
    """

    def __init__(
        self,
        clips: List[List[int]],
        shuffle: bool = True,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        if num_replicas is None or rank is None:
            from mmcv.runner import get_dist_info

            _rank, _world = get_dist_info()
            num_replicas = num_replicas if num_replicas is not None else _world
            rank = rank if rank is not None else _rank

        self.clips = clips
        self.shuffle = shuffle
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self._epoch = 0

        # Pad clip count so every rank gets the same number of batches (keeps
        # DDP in lock-step).  Padding repeats earlier clips; harmless for train,
        # and for eval num_replicas is 1 so no padding happens.
        self.num_clips_per_rank = int(math.ceil(len(self.clips) / self.num_replicas))
        self.total_clips = self.num_clips_per_rank * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self) -> Iterator[List[int]]:
        order = list(range(len(self.clips)))
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self._epoch)
            rng.shuffle(order)
            # Advance epoch so a fresh iterator (next epoch) reshuffles even
            # without an external set_epoch hook.
            self._epoch += 1

        if self.total_clips > len(order):
            order = order + order[: self.total_clips - len(order)]

        # Shard across ranks.
        order = order[self.rank : self.total_clips : self.num_replicas]

        for clip_idx in order:
            yield list(self.clips[clip_idx])

    def __len__(self) -> int:
        return self.num_clips_per_rank
