import multiprocessing as mp
import pickle
from os import path as osp

import mmcv
import numpy as np
from mmcv import track_iter_progress
from mmcv.ops import roi_align
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO

from mmdet3d.core.bbox import box_np_ops as box_np_ops
from mmdet3d.datasets import build_dataset
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps

# Module-level globals for worker processes
_dataset_worker = None
_database_save_path_worker = None
_info_prefix_worker = None
_used_classes_worker = None


def _poly2mask(mask_ann, img_h, img_w):
    if isinstance(mask_ann, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        rle = maskUtils.merge(rles)
    elif isinstance(mask_ann["counts"], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
    else:
        # rle
        rle = mask_ann
    mask = maskUtils.decode(rle)
    return mask


def _parse_coco_ann_info(ann_info):
    gt_bboxes = []
    gt_labels = []
    gt_bboxes_ignore = []
    gt_masks_ann = []

    for i, ann in enumerate(ann_info):
        if ann.get("ignore", False):
            continue
        x1, y1, w, h = ann["bbox"]
        if ann["area"] <= 0:
            continue
        bbox = [x1, y1, x1 + w, y1 + h]
        if ann.get("iscrowd", False):
            gt_bboxes_ignore.append(bbox)
        else:
            gt_bboxes.append(bbox)
            gt_masks_ann.append(ann["segmentation"])

    if gt_bboxes:
        gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        gt_labels = np.array(gt_labels, dtype=np.int64)
    else:
        gt_bboxes = np.zeros((0, 4), dtype=np.float32)
        gt_labels = np.array([], dtype=np.int64)

    if gt_bboxes_ignore:
        gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
    else:
        gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

    ann = dict(bboxes=gt_bboxes, bboxes_ignore=gt_bboxes_ignore, masks=gt_masks_ann)

    return ann


def crop_image_patch_v2(pos_proposals, pos_assigned_gt_inds, gt_masks):
    import torch
    from torch.nn.modules.utils import _pair

    device = pos_proposals.device
    num_pos = pos_proposals.size(0)
    fake_inds = torch.arange(num_pos, device=device).to(dtype=pos_proposals.dtype)[
        :, None
    ]
    rois = torch.cat([fake_inds, pos_proposals], dim=1)  # Nx5
    mask_size = _pair(28)
    rois = rois.to(device=device)
    gt_masks_th = (
        torch.from_numpy(gt_masks)
        .to(device)
        .index_select(0, pos_assigned_gt_inds)
        .to(dtype=rois.dtype)
    )
    # Use RoIAlign could apparently accelerate the training (~0.1s/iter)
    targets = roi_align(gt_masks_th, rois, mask_size[::-1], 1.0, 0, True).squeeze(1)
    return targets


def crop_image_patch(pos_proposals, gt_masks, pos_assigned_gt_inds, org_img):
    num_pos = pos_proposals.shape[0]
    masks = []
    img_patches = []
    for i in range(num_pos):
        gt_mask = gt_masks[pos_assigned_gt_inds[i]]
        bbox = pos_proposals[i, :].astype(np.int32)
        x1, y1, x2, y2 = bbox
        w = np.maximum(x2 - x1 + 1, 1)
        h = np.maximum(y2 - y1 + 1, 1)

        mask_patch = gt_mask[y1 : y1 + h, x1 : x1 + w]
        masked_img = gt_mask[..., None] * org_img
        img_patch = masked_img[y1 : y1 + h, x1 : x1 + w]

        img_patches.append(img_patch)
        masks.append(mask_patch)
    return img_patches, masks


def _process_one_gt_sample(j):
    """Worker: extract and write GT objects for one dataset sample, return db_info entries."""
    dataset = _dataset_worker
    database_save_path = _database_save_path_worker
    info_prefix = _info_prefix_worker
    used_classes = _used_classes_worker

    input_dict = dataset.get_data_info(j)
    dataset.pre_pipeline(input_dict)
    example = dataset.pipeline(input_dict)
    annos = example['ann_info']
    image_idx = example['sample_idx']
    points = example['points'].tensor.numpy()
    gt_boxes_3d = annos['gt_bboxes_3d'].tensor.numpy()
    names = annos['gt_names']

    group_ids = (
        annos['group_ids']
        if 'group_ids' in annos
        else np.arange(gt_boxes_3d.shape[0], dtype=np.int64)
    )
    difficulty = annos.get('difficulty', np.zeros(gt_boxes_3d.shape[0], dtype=np.int32))

    num_obj = gt_boxes_3d.shape[0]
    point_indices = box_np_ops.points_in_rbbox(points, gt_boxes_3d)

    local_db_infos = []  # list of (name, db_info, local_group_id)
    for i in range(num_obj):
        filename = f'{image_idx}_{names[i]}_{i}.bin'
        abs_filepath = osp.join(database_save_path, filename)
        rel_filepath = osp.join(f'{info_prefix}_gt_database', filename)

        gt_points = points[point_indices[:, i]]
        gt_points[:, :3] -= gt_boxes_3d[i, :3]

        with open(abs_filepath, 'wb') as f:
            gt_points.tofile(f)

        if (used_classes is None) or names[i] in used_classes:
            db_info = {
                'name': names[i],
                'path': rel_filepath,
                'image_idx': image_idx,
                'gt_idx': i,
                'box3d_lidar': gt_boxes_3d[i],
                'num_points_in_gt': gt_points.shape[0],
                'difficulty': difficulty[i],
            }
            if 'score' in annos:
                db_info['score'] = annos['score'][i]
            local_db_infos.append((names[i], db_info, int(group_ids[i])))

    return local_db_infos


def create_groundtruth_database(
    dataset_class_name,
    data_path,
    info_prefix,
    info_path=None,
    mask_anno_path=None,
    used_classes=None,
    database_save_path=None,
    db_info_save_path=None,
    relative_path=True,
    add_rgb=False,
    lidar_only=False,
    bev_only=False,
    coors_range=None,
    with_mask=False,
    load_augmented=None,
    num_workers=4,
):
    """Given the raw data, generate the ground truth database.

    Args:
        dataset_class_name （str): Name of the input dataset.
        data_path (str): Path of the data.
        info_prefix (str): Prefix of the info file.
        info_path (str): Path of the info file.
            Default: None.
        mask_anno_path (str): Path of the mask_anno.
            Default: None.
        used_classes (list[str]): Classes have been used.
            Default: None.
        database_save_path (str): Path to save database.
            Default: None.
        db_info_save_path (str): Path to save db_info.
            Default: None.
        relative_path (bool): Whether to use relative path.
            Default: True.
        with_mask (bool): Whether to use mask.
            Default: False.
    """
    print(f"Create GT Database of {dataset_class_name}")
    dataset_cfg = dict(
        type=dataset_class_name, dataset_root=data_path, ann_file=info_path
    )
    if dataset_class_name == "KittiDataset":
        dataset_cfg.update(
            test_mode=False,
            split="training",
            modality=dict(
                use_lidar=True,
                use_depth=False,
                use_lidar_intensity=True,
                use_camera=with_mask,
            ),
            pipeline=[
                dict(
                    type="LoadPointsFromFile",
                    coord_type="LIDAR",
                    load_dim=4,
                    use_dim=4,
                ),
                dict(
                    type="LoadAnnotations3D",
                    with_bbox_3d=True,
                    with_label_3d=True,
                ),
            ],
        )

    elif dataset_class_name == "NuScenesDataset":
        if not load_augmented:
            dataset_cfg.update(
                use_valid_flag=True,
                pipeline=[
                    dict(
                        type="LoadPointsFromFile",
                        coord_type="LIDAR",
                        load_dim=5,
                        use_dim=5,
                    ),
                    dict(
                        type="LoadPointsFromMultiSweeps",
                        sweeps_num=10,
                        use_dim=[0, 1, 2, 3, 4],
                        pad_empty_sweeps=True,
                        remove_close=True,
                    ),
                    dict(
                        type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True
                    ),
                ],
            )
        else:
            dataset_cfg.update(
                use_valid_flag=True,
                pipeline=[
                    dict(
                        type="LoadPointsFromFile",
                        coord_type="LIDAR",
                        load_dim=16,
                        use_dim=list(range(16)),
                        load_augmented=load_augmented,
                    ),
                    dict(
                        type="LoadPointsFromMultiSweeps",
                        sweeps_num=10,
                        load_dim=16,
                        use_dim=list(range(16)),
                        pad_empty_sweeps=True,
                        remove_close=True,
                        load_augmented=load_augmented,
                    ),
                    dict(
                        type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True
                    ),
                ],
            )

    elif dataset_class_name == "WaymoDataset":
        dataset_cfg.update(
            test_mode=False,
            split="training",
            modality=dict(
                use_lidar=True,
                use_depth=False,
                use_lidar_intensity=True,
                use_camera=False,
            ),
            pipeline=[
                dict(
                    type="LoadPointsFromFile",
                    coord_type="LIDAR",
                    load_dim=6,
                    use_dim=5,
                ),
                dict(
                    type="LoadAnnotations3D",
                    with_bbox_3d=True,
                    with_label_3d=True,
                ),
            ],
        )

    dataset = build_dataset(dataset_cfg)

    if database_save_path is None:
        database_save_path = osp.join(data_path, f"{info_prefix}_gt_database")
    if db_info_save_path is None:
        db_info_save_path = osp.join(data_path, f"{info_prefix}_dbinfos_train.pkl")
    mmcv.mkdir_or_exist(database_save_path)

    global _dataset_worker, _database_save_path_worker, _info_prefix_worker, _used_classes_worker
    _dataset_worker = dataset
    _database_save_path_worker = database_save_path
    _info_prefix_worker = info_prefix
    _used_classes_worker = used_classes

    indices = list(range(len(dataset)))
    if num_workers > 1:
        ctx = mp.get_context('fork')
        with ctx.Pool(num_workers) as pool:
            prog = mmcv.ProgressBar(len(indices))
            all_results = []
            for r in pool.imap(_process_one_gt_sample, indices, chunksize=8):
                all_results.append(r)
                prog.update()
            print()
    else:
        all_results = [
            _process_one_gt_sample(j)
            for j in track_iter_progress(indices)
        ]

    # Accumulate results and assign globally unique group_ids
    all_db_infos = dict()
    group_counter = 0
    for sample_entries in all_results:
        group_remap = {}
        for name, db_info, local_group_id in sample_entries:
            if local_group_id not in group_remap:
                group_remap[local_group_id] = group_counter
                group_counter += 1
            db_info['group_id'] = group_remap[local_group_id]
            if name in all_db_infos:
                all_db_infos[name].append(db_info)
            else:
                all_db_infos[name] = [db_info]

    for k, v in all_db_infos.items():
        print(f'load {len(v)} {k} database infos')

    with open(db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)
