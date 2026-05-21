import os
from collections import OrderedDict

import mmcv
from mmdet.datasets import DATASETS

from .nuscenes_dataset import NuScenesDataset

__all__ = ["TemporalNuScenesDataset"]


@DATASETS.register_module()
class TemporalNuScenesDataset(NuScenesDataset):
    """NuScenes dataset that exposes temporal *clips* for ConvGRU training.

    Behaves exactly like :class:`NuScenesDataset` at the per-sample level
    (``__getitem__`` returns one frame), but additionally builds ``self.clips``
    -- a list of clips, each an ordered list of dataset indices drawn from a
    single nuScenes scene.  :class:`ClipBatchSampler` consumes ``self.clips`` so
    that each dataloader batch is one ordered clip (batch dim == time), which
    :class:`ConvGRUFuser` fuses recurrently.

    Scene membership and intra-scene ordering are resolved with the nuScenes
    devkit (``sample['scene_token']``); the result is cached next to the ann
    file so repeated runs don't reload the devkit tables.

    Clip construction:
      * training (``test_mode=False``): dense sliding windows of length
        ``clip_length`` with stride ``clip_stride`` (default 1).  Scenes shorter
        than ``clip_length`` are skipped.
      * evaluation (``test_mode=True``): a non-overlapping *partition* of each
        scene into windows of up to ``clip_length`` frames.  Because the base
        infos are globally timestamp-sorted (scenes are contiguous), visiting
        these clips in order traverses dataset indices 0..N-1 exactly once, so
        the standard evaluation result<->info indexing stays valid.

    Args:
        clip_length: number of frames T per clip.
        clip_stride: stride between training windows (ignored in test mode).
    """

    def __init__(self, *args, clip_length: int = 3, clip_stride: int = 1, **kwargs):
        # torchpack config merge is a non-deleting deep-merge, so overriding the
        # base CBGS `data.train` leaves a stray `dataset:` sub-config behind.
        # It is redundant with the fields set here, so drop it.
        kwargs.pop("dataset", None)
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        super().__init__(*args, **kwargs)
        self.clips = self._build_clips()

    # ------------------------------------------------------------------
    def _scene_tokens(self):
        """Return scene_token for each data_info index, using a cached map."""
        tokens = [info["token"] for info in self.data_infos]

        cache_path = None
        if isinstance(self.ann_file, str):
            cache_path = self.ann_file + ".scene_map.pkl"

        token2scene = None
        if cache_path is not None and os.path.exists(cache_path):
            cached = mmcv.load(cache_path)
            if all(t in cached for t in tokens):
                token2scene = cached

        if token2scene is None:
            from nuscenes import NuScenes

            nusc = NuScenes(
                version=self.version, dataroot=self.dataset_root, verbose=False
            )
            token2scene = {
                s["token"]: s["scene_token"] for s in nusc.sample
            }
            if cache_path is not None:
                try:
                    mmcv.dump(token2scene, cache_path)
                except OSError:
                    pass
            del nusc

        return [token2scene[t] for t in tokens]

    def _build_clips(self):
        scene_tokens = self._scene_tokens()

        # Group indices by scene, preserving first-appearance (== timestamp)
        # order so the concatenation of clips stays globally ordered.
        groups = OrderedDict()
        for idx, scene in enumerate(scene_tokens):
            groups.setdefault(scene, []).append(idx)

        clips = []
        T = self.clip_length
        if self.test_mode:
            # Non-overlapping partition covering every frame once.
            for idxs in groups.values():
                for s in range(0, len(idxs), T):
                    clips.append(idxs[s : s + T])
        else:
            # Dense sliding windows of exactly T frames.
            stride = max(1, self.clip_stride)
            for idxs in groups.values():
                if len(idxs) < T:
                    continue
                for s in range(0, len(idxs) - T + 1, stride):
                    clips.append(idxs[s : s + T])

        if len(clips) == 0:
            raise RuntimeError(
                "TemporalNuScenesDataset built 0 clips; check clip_length "
                f"({T}) against scene lengths."
            )
        return clips
