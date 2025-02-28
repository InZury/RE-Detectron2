from re_detectron2 import CFile


def pairwise_iou_rotated(boxes1, boxes2):
    return CFile.box_iou_rotated(boxes1, boxes2)
