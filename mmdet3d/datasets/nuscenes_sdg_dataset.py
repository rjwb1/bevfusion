from typing import Any, Dict

import numpy as np

from mmdet.datasets import DATASETS
from .nuscenes_dataset import NuScenesDataset


@DATASETS.register_module()
class NuScenesSDGDataset(NuScenesDataset):
    """NuScenesDataset variant for synthetic (Isaac SDG) vineyard data.

    Two differences from the base class:

    1. ``get_data_info`` forwards the per-sample BEV-segmentation *geometry*
       (``bev_rows`` / ``bev_obstacles``, ego frame) from the info pkl into the
       pipeline input dict, so ``LoadBEVSegmentationFromGeometry`` can rasterise
       rows/obstacles instead of querying the (non-existent) nuScenes map API.

    2. ``evaluate`` is a no-op by default: the nuScenes detection devkit needs a
       real nuScenes release + map to score against and would crash on synthetic
       locations. Training/validation still runs; wire in a custom metric when
       you want numbers. Pass ``evaluate_detection=True`` to fall back to the
       base nuScenes evaluation (only valid if your data is devkit-loadable).
    """

    def __init__(
        self,
        *args,
        evaluate_detection: bool = False,
        keep_zero_point_boxes: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.evaluate_detection = evaluate_detection
        # Synthetic labels are trustworthy even when the (sparse) synthetic LiDAR
        # returns no points on an object. When True, keep every annotated box
        # instead of filtering on num_lidar_pts/valid_flag (base get_ann_info).
        self.keep_zero_point_boxes = keep_zero_point_boxes

    def get_ann_info(self, index: int) -> Dict[str, Any]:
        if not self.keep_zero_point_boxes:
            return super().get_ann_info(index)
        # Present an all-True valid_flag so the base method's point filter keeps
        # every box, then restore the info dict so nothing else is affected.
        info = self.data_infos[index]
        saved_flag = info.get("valid_flag")
        saved_uvf = self.use_valid_flag
        info["valid_flag"] = np.ones(len(info["gt_boxes"]), dtype=bool)
        self.use_valid_flag = True
        try:
            return super().get_ann_info(index)
        finally:
            self.use_valid_flag = saved_uvf
            if saved_flag is None:
                del info["valid_flag"]
            else:
                info["valid_flag"] = saved_flag

    def get_data_info(self, index: int) -> Dict[str, Any]:
        data = super().get_data_info(index)
        info = self.data_infos[index]
        # Ego/LiDAR-frame vector geometry for the BEV-seg target. Default to
        # empty arrays so the transform produces all-background masks rather
        # than failing if a sample has no rows/obstacles.
        data["bev_rows"] = np.asarray(
            info.get("bev_rows", np.zeros((0, 2, 2))), dtype=np.float32
        )
        data["bev_obstacles"] = info.get("bev_obstacles", [])
        return data

    def evaluate(self, results, *args, **kwargs):
        if self.evaluate_detection:
            return super().evaluate(results, *args, **kwargs)
        import logging

        logging.getLogger(__name__).warning(
            "NuScenesSDGDataset.evaluate is a no-op (synthetic data has no "
            "nuScenes devkit GT). Returning empty metrics."
        )
        return {}
