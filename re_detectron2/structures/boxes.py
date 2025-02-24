import math
import torch
import numpy as np

from enum import IntEnum, unique
from typing import Any, List, Tuple, Union

RawBoxType = Union[List[float], Tuple[float, ...], torch.Tensor, np.ndarray]


@unique
class BoxMode(IntEnum):
    XYXY_ABS = 0  # (x0, y0, x1, y1) to absolute
    XYWH_ABS = 1  # (x0, y0, width, height) to absolute
    XYXY_REL = 2  # (x0, y0, x1, y1) to relative
    XYWH_REL = 3  # (x0, y0, width, height) to relative
    XYWHA_ABS = 4  # (xc, yc, width, height, angle) to absolute, xc and yc are center of box

    @staticmethod
    def convert(box: RawBoxType, from_mode: "BoxMode", to_mode: "BoxMode") -> RawBoxType:
        if from_mode == to_mode:
            return box

        original_type = type(box)
        is_numpy = isinstance(box, np.ndarray)
        single_box = isinstance(box, (list, tuple))

        if single_box:
            assert len(box) == 4 or len(box) == 5, (
                "BoxMode.convert takes either a k-tuple/list or an Nxk array/tensor, where k == 4 or 5"
            )

            arr = torch.tensor(box)[None, :]
        else:
            if is_numpy:
                arr = torch.from_numpy(np.asarray(box)).clone()
            else:
                arr = box.clone()

        assert (to_mode.value not in [BoxMode.XYXY_REL, BoxMode.XYWH_REL] and
                from_mode.value not in [BoxMode.XYXY_REL, BoxMode.XYWH_REL]), (
            "Relative mode not yet supported!"
        )

        if from_mode == BoxMode.XYWHA_ABS and to_mode == BoxMode.XYXY_ABS:
            assert (
                arr.shape[-1] == 5
            ), "The last dimension of input shape must be 5 for XYWHA format"

            original_dtype = arr.dtype
            arr = arr.double()

            width = arr[:, 2]
            height = arr[:, 3]
            angle = arr[:, 4]
            cos = torch.abs(torch.cos(angle * math.pi / 180.0))
            sin = torch.abs(torch.sin(angle * math.pi / 180.0))

            new_width = cos * width + sin * height
            new_height = cos * height + sin * width

            arr[:, 0] -= new_width / 2.0
            arr[:, 1] -= new_height / 2.0
            arr[:, 2] = arr[:, 0] + new_width
            arr[:, 3] = arr[:, 1] + new_height

            arr = arr[:, :4].to(dtype=original_dtype)
        elif from_mode == BoxMode.XYWH_ABS and to_mode == BoxMode.XYWHA_ABS:
            original_dtype = arr.dtype
            arr = arr.double()

            arr[:, 0] += arr[:, 2] / 2.0
            arr[:, 1] += arr[:, 3] / 2.0

            angles = torch.zeros((arr.shape[0], 1), dtype=arr.dtype)
            arr = torch.cat((arr, angles), dim=1).to(dtype=original_dtype)
        else:
            if from_mode == BoxMode.XYWH_ABS and to_mode == BoxMode.XYXY_ABS:
                arr[:, 2] += arr[:, 0]
                arr[:, 3] += arr[:, 1]
            elif from_mode == BoxMode.XYXY_ABS and to_mode == BoxMode.XYWH_ABS:
                arr[:, 2] -= arr[:, 0]
                arr[:, 3] -= arr[:, 1]
            else:
                raise NotImplementedError(f"Conversion from BoxMode {from_mode} to {to_mode} is not supported yet")

        if single_box:
            return original_type(arr.flatten().tolist())
        if is_numpy:
            return arr.numpy()
        else:
            return arr


class Boxes:
    def __init__(self, tensor: torch.Tensor):
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)

        if tensor.numel() == 0:
            tensor = tensor.reshape((0, 4)).to(dtype=torch.float32, device=device)

        assert tensor.dim() == 2 and tensor.size(-1) == 4, tensor.size()

        self.tensor = tensor

    def clone(self) -> "Boxes":
        return Boxes(self.tensor.clone())

    @torch.jit.unused
    def to(self, *args: Any, **kwargs: Any):
        return Boxes(self.tensor.to(*args, **kwargs))

    def area(self) -> torch.Tensor:
        box = self.tensor
        area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])

        return area

    def clip(self, box_size: Tuple[int, int]) -> None:
        assert torch.isfinite(self.tensor).all(), "Box tensor contains infinite or NaN!"

        height, width = box_size

        self.tensor[:, 0].clamp_(min=0, max=width)
        self.tensor[:, 1].clamp_(min=0, max=height)
        self.tensor[:, 2].clamp_(min=0, max=width)
        self.tensor[:, 3].clamp_(min=0, max=height)

    def nonempty(self, threshold: float = 0.0) -> torch.Tensor:
        box = self.tensor
        width = box[:, 2] - box[:, 0]
        height = box[:, 3] - box[:, 1]
        keep = (width > threshold) & (height > threshold)

        return keep

    def __getitem__(self, item):
        if isinstance(item, int):
            return Boxes(self.tensor[item].view(1, -1))

        box = self.tensor[item]

        assert box.dim() == 2, f"Indexing on Boxes with {item} failed to return a matrix!"

        return Boxes(box)

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        return f"Boxes({self.tensor})"

    def inside_box(self, box_size: Tuple[int, int], boundary_threshold: int = 0) -> torch.Tensor:
        height, width = box_size

        return (
            (self.tensor[..., 0] >= -boundary_threshold)
            & (self.tensor[..., 1] >= -boundary_threshold)
            & (self.tensor[..., 2] < width + boundary_threshold)
            & (self.tensor[..., 3] < height + boundary_threshold)
        )

    def get_center(self) -> torch.Tensor:
        return (self.tensor[:, :2] + self.tensor[:, 2:]) / 2

    def scale(self, scale_x: float, scale_y: float) -> None:
        self.tensor[:, 0::2] *= scale_x
        self.tensor[:, 1::2] *= scale_y

    @classmethod
    @torch.jit.unused
    def cat(cls, boxes_list):
        assert isinstance(boxes_list, (list, tuple))

        if len(boxes_list) == 0:
            return cls(torch.empty(0))

        assert all([isinstance(box, Boxes) for box in boxes_list])

        return cls(torch.cat([box.tensor for box in boxes_list], dim=0))

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    @torch.jit.unused
    def __iter__(self):
        yield from self.tensor


def pairwise_intersection(boxes_1: Boxes, boxes_2: Boxes) -> torch.Tensor:
    boxes_1, boxes_2 = boxes_1.tensor, boxes_2.tensor
    width_height = (torch.min(boxes_1[:, None, 2:], boxes_2[:, 2:]) -
                    torch.max(boxes_1[:, None, :2], boxes_2[:, :2]))

    width_height.clamp_(min=0)
    intersection = width_height.prod(dim=2)

    return intersection


def pairwise_iou(boxes_1: Boxes, boxes_2: Boxes) -> torch.Tensor:
    area_1 = boxes_1.area()
    area_2 = boxes_2.area()
    intersection = pairwise_intersection(boxes_1, boxes_2)

    return torch.where(
        intersection > 0,
        intersection / (area_1[:, None] + area_2 - intersection),
        torch.zeros(1, dtype=intersection.dtype, device=intersection.device)
    )


def pairwise_ioa(boxes_1: Boxes, boxes_2: Boxes) -> torch.Tensor:
    area_2 = boxes_2.area()
    intersection = pairwise_intersection(boxes_1, boxes_2)

    return torch.where(
        intersection > 0,
        intersection / area_2,
        torch.zeros(1, dtype=intersection.dtype, device=intersection.device)
    )


def matched_box_list_iou(boxes_1: Boxes, boxes_2: Boxes) -> torch.Tensor:
    assert (len(boxes_1) == len(boxes_2)
            ), f"box lists should have the same number of entries, got {len(boxes_1)}, {len(boxes_2)}"

    area_1 = boxes_1.area()
    area_2 = boxes_2.area()
    box_1, box_2 = boxes_1.tensor, boxes_2.tensor
    left_top = torch.max(box_1[:, :2], box_2[:, :2])
    right_bottom = torch.min(box_1[:, 2:], box_2[:, 2:])
    width_height = (right_bottom - left_top).clamp_(min=0)
    intersection = width_height[:, 0] * width_height[:, 1]

    return intersection / (area_1 + area_2 - intersection)
