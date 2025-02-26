# -*- coding: utf-8 -*-
import torch

from typing import List
from torchvision.ops import boxes as boxes_ops
from torchvision.ops import nms


# NMS(Non Maximum Suppression)
def batched_nms(
        boxes: torch.Tensor, scores: torch.Tensor, indexes: torch.Tensor, iou_threshold: float
):
    assert boxes.shape[-1] == 4

    if len(boxes) < 40000:
        return boxes_ops.batched_nms(boxes, scores, indexes, iou_threshold)

    result_mask = scores.new_zeros(scores.size(), dtype=torch.bool)

    for index in torch.jit.annotate(List[int], torch.unique(indexes).cpu().tolist()):
        mask = (indexes == index).nonzero().view(-1)
        keep = nms(boxes[mask], scores[mask], iou_threshold)
        result_mask[mask[keep]] = True

    keep = result_mask.nonzero().view(-1)
    keep = keep[scores[keep].argsort(descending=True)]

    return keep


def nmp_rotated(boxes, scores, iou_threshold):
    # import CONFIG
    # return CONFIG.nms_rotated(boxes, scores, iou_threshold)

    return None


def batched_nms_rotated(boxes, scores, indexes, iou_threshold):
    assert boxes.shape[-1] == 5

    if boxes.numel() == 0:
        return torch.empty((0, ), dtype=torch.int64, device=boxes.device)

    max_coordinate = (torch.max(boxes[:, 0], boxes[:, 1]) + torch.max(boxes[:, 2], boxes[:, 3]) / 2).max()
    min_coordinate = (torch.max(boxes[:, 0], boxes[:, 1]) - torch.max(boxes[:, 2], boxes[:, 3]) / 2).min()
    offsets = indexes.to(boxes) * (max_coordinate - min_coordinate + 1)
    boxes_for_nms = boxes.clone()
    boxes_for_nms[:, :2] += offsets[:, None]
    keep = nmp_rotated(boxes_for_nms, scores, iou_threshold)

    return keep
