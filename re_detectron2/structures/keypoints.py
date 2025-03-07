import torch
import numpy as np

from typing import Any, List, Tuple, Union

from ..layers import interpolate


class Keypoints:
    def __init__(self, keypoints: Union[torch.Tensor, np.ndarray, List[torch.Tensor], List[List[float]]]):
        device = keypoints.device if isinstance(keypoints, torch.Tensor) else torch.device("cpu")
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32, device=device)

        assert keypoints.dim() == 3 and keypoints.shape[2] == 3, keypoints.shape

        self.tensor = keypoints

    def __len__(self) -> int:
        return self.tensor.size(0)

    def to(self, *args: Any, **kwargs: Any) -> "Keypoints":
        return type(self)(self.tensor.to(*args, **kwargs))

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    def to_heatmap(self, boxes: torch.Tensor, heatmap_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return keypoint_to_heatmap(self.tensor, boxes, heatmap_size)

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Keypoints":
        if isinstance(item, int):
            return Keypoints([self.tensor[item]])

        return Keypoints(self.tensor[item])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_instances={len(self.tensor)})"


def keypoint_to_heatmap(
        keypoints: torch.Tensor, rois: torch.Tensor, heatmap_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    if rois.numel() == 0:
        return rois.new().long(), rois.new().long()

    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    scale_x = heatmap_size / (rois[:, 2] - rois[:, 0])
    scale_y = heatmap_size / (rois[:, 3] - rois[:, 1])

    offset_x = offset_x[:, None]
    offset_y = offset_y[:, None]
    scale_x = scale_x[:, None]
    scale_y = scale_y[:, None]

    x = keypoints[..., 0]
    y = keypoints[..., 1]

    x_boundary_indexes = x == rois[:, 2][:, None]
    y_boundary_indexes = y == rois[:, 3][:, None]

    x = (x - offset_x) * scale_x
    x = x.floor().long()
    y = (y - offset_y) * scale_y
    y = y.floor().long()

    x[x_boundary_indexes] = heatmap_size - 1
    y[y_boundary_indexes] = heatmap_size - 1

    valid_loc = (x >= 0) & (y >= 0) & (x < heatmap_size) & (y < heatmap_size)
    visual = keypoints[..., 2] > 0
    valid = (valid_loc & visual).long()

    linear_index = y * heatmap_size + x
    heatmaps = linear_index * valid

    return heatmaps, valid


def heatmaps_to_keypoints(maps: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
    maps = maps.detach()
    rois = rois.detach()

    offset_x = rois[:, 0]
    offset_y = rois[:, 1]

    widths = (rois[:, 2] - rois[:, 0]).clamp(min=1)
    heights = (rois[:, 3] - rois[:, 1]).clamp(min=1)
    widths_ceil = widths.ceil()
    heights_ceil = heights.ceil()

    num_rois, num_keypoints = maps.shape[:2]
    xy_predict = maps.new_zeros(rois.shape[0], num_keypoints, 4)

    width_corrections = widths / widths_ceil
    height_corrections = heights / heights_ceil

    keypoints_index = torch.arange(num_keypoints, device=maps.device)

    for index in range(num_rois):
        outsize = (int(heights_ceil[index]), int(widths_ceil[index]))
        roi_map = interpolate(maps[[index]], size=outsize, mode="bicubic", align_corners=False).squeeze(0)
        max_score, _ = roi_map.view(num_keypoints, -1).max(1)
        max_score = max_score.view(num_keypoints, 1, 1)
        temp_full_resolution = (roi_map - max_score).exp_()
        temp_pool_resolution = (maps[index] - max_score).exp_()
        roi_map_scores = temp_full_resolution / temp_pool_resolution.sum((1, 2), keepdims=True)

        width = roi_map.shape[2]
        pos = roi_map.view(num_keypoints, -1).argmax(1)

        x_int = pos % width
        y_int = (pos - x_int) // width

        assert (
            torch.Tensor(roi_map_scores[keypoints_index, y_int, x_int]
                         == roi_map_scores.view(num_keypoints, -1).max(1)[0])
        ).all()

        x = (x_int.float() + 0.5) * width_corrections[index]
        y = (y_int.float() + 0.5) * height_corrections[index]

        xy_predict[index, :, 0] = x + offset_x[index]
        xy_predict[index, :, 1] = y + offset_y[index]
        xy_predict[index, :, 2] = roi_map[keypoints_index, y_int, x_int]
        xy_predict[index, :, 3] = roi_map_scores[keypoints_index, y_int, x_int]

    return xy_predict
