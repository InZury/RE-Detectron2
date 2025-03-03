import math
from typing import Any, Iterator, Tuple, Union

import torch

from .boxes import Boxes
from ..layers.rotated_boxes import pairwise_iou_rotated


class RotatedBoxes(Boxes):
    def __init__(self, tensor: torch.Tensor):
        super().__init__(tensor)
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)

        if tensor.numel() == 0:
            tensor = tensor.reshape((0, 5)).to(dtype=torch.float32, device=device)

        assert tensor.dim() == 2 and tensor.size(-1) == 5, tensor.size()

        self.tensor = tensor

    def clone(self) -> "RotatedBoxes":
        return RotatedBoxes(self.tensor.clone())

    def to(self, *args: Any, **kwargs: Any) -> "RotatedBoxes":
        return RotatedBoxes(self.tensor.to(*args, **kwargs))

    def area(self) -> torch.Tensor:
        box = self.tensor
        area = box[:, 2] * box[:, 3]

        return area

    def normalize_angles(self) -> None:
        self.tensor[:, 4] = (self.tensor[:, 4] + 180.0) % 360.0 - 180.0

    def clip(self, box_size: Tuple[int, int], clip_angle_threshold: float = 1.0) -> None:
        height, width = box_size

        self.normalize_angles()
        index = torch.where(torch.abs(self.tensor[:, 4]) <= clip_angle_threshold)[0]
        x1 = self.tensor[index, 0] - self.tensor[index, 2] / 2.0
        y1 = self.tensor[index, 1] - self.tensor[index, 3] / 2.0
        x2 = self.tensor[index, 0] + self.tensor[index, 2] / 2.0
        y2 = self.tensor[index, 1] + self.tensor[index, 3] / 2.0

        x1.clamp_(min=0, max=width)
        y1.clamp_(min=0, max=height)
        x2.clamp_(min=0, max=width)
        y2.clamp_(min=0, max=height)

        self.tensor[index, 0] = (x1 + x2) / 2.0
        self.tensor[index, 1] = (y1 + y2) / 2.0
        self.tensor[index, 2] = torch.min(self.tensor[index, 2], x2 - x1)
        self.tensor[index, 3] = torch.min(self.tensor[index, 3], y2 - y1)

    def nonempty(self, threshold: float = 0.0) -> torch.Tensor:
        box = self.tensor
        widths = box[:, 2]
        heights = box[:, 3]
        keep = (widths > threshold) & (heights > threshold)

        return keep

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "RotatedBoxes":
        if isinstance(item, int):
            return RotatedBoxes(self.tensor[item].view(1, -1))

        box = self.tensor[item]

        assert box.dim() == 2, f"Indexing on Rotated Boxes with {item} failed to return a matrix!"

        return RotatedBoxes(box)

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        return f"RotatedBoxes({self.tensor})"

    def inside_box(self, box_size: Tuple[int, int], boundary_threshold: int = 0) -> torch.Tensor:
        height, width = box_size
        center_x = self.tensor[..., 0]
        center_y = self.tensor[..., 1]
        half_width = self.tensor[..., 2] / 2.0
        half_height = self.tensor[..., 3] / 2.0
        angle = self.tensor[..., 4]
        cos = torch.abs(torch.cos(angle * math.pi / 180.0))
        sin = torch.abs(torch.sin(angle * math.pi / 180.0))
        max_rect_distance_x = cos * half_width + sin * half_height
        max_rect_distance_y = cos * half_height + sin * half_width

        indexes_inside = (
            (center_x - max_rect_distance_x >= -boundary_threshold)
            & (center_y - max_rect_distance_y >= -boundary_threshold)
            & (center_x + max_rect_distance_x < width + boundary_threshold)
            & (center_y + max_rect_distance_y < height + boundary_threshold)
        )

        return indexes_inside

    def get_centers(self) -> torch.Tensor:
        return self.tensor[:, :2]

    def scale(self, scale_x: float, scale_y: float) -> None:
        self.tensor[:, 0] *= scale_x
        self.tensor[:, 1] *= scale_y

        theta = self.tensor[:, 4] * math.pi / 180.0
        cos = torch.abs(theta)
        sin = torch.abs(theta)

        self.tensor[:, 2] *= torch.sqrt((scale_x * cos) ** 2 + (scale_y * sin) ** 2)
        self.tensor[:, 3] *= torch.sqrt((scale_x * sin) ** 2 + (scale_y * cos) ** 2)
        self.tensor[:, 4] = torch.atan2(scale_x * sin, scale_y * cos) * 180.0 / math.pi

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    def __iter__(self) -> Iterator[torch.Tensor]:
        yield from self.tensor


def pairwise_iou(boxes1: RotatedBoxes, boxes2: RotatedBoxes) -> None:
    return pairwise_iou_rotated(boxes1.tensor, boxes2.tensor)
