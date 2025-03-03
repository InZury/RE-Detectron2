import copy
import itertools
import torch
import numpy as np
import pycocotools.mask as mask_util

from typing import Any, Iterator, List, Union

from ..layers.roi_align import ROIAlign
from .boxes import Boxes


class BitMasks:
    def __init__(self, tensor: Union[torch.Tensor, np.ndarray]):
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.bool, device=device)

        assert tensor.dim() == 3, tensor.size()
        self.image_size = tensor.shape[1:]
        self.tensor = tensor

    def to(self, *args: Any, **kwargs: Any) -> "BitMasks":
        return BitMasks(self.tensor.to(*args, **kwargs))

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "BitMasks":
        if isinstance(item, int):
            return BitMasks(self.tensor[item].view(1, -1))

        mask = self.tensor[item]

        assert mask.dim() == 3, f"Indexing on Bitmasks with {item} returns a tensor with shape {mask.shape}!"

        return BitMasks(mask)

    def __iter__(self) -> torch.Tensor:
        yield from self.tensor

    def __repr__(self) -> str:
        string = f"num_instances={self.tensor}"

        return f"{self.__class__.__name__}({string})"

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def nonempty(self) -> torch.Tensor:
        return self.tensor.flatten(1).any(dim=1)

    @staticmethod
    def from_polygon_masks(
            polygon_masks: Union["PolygonMasks", List[List[np.ndarray]]], height: int, width: int
    ) -> "BitMasks":
        if isinstance(polygon_masks, PolygonMasks):
            polygon_masks = polygon_masks.polygons

        masks = [polygon_to_bitmask(polygon, height, width) for polygon in polygon_masks]

        return BitMasks(torch.stack([torch.from_numpy(x) for x in masks]))

    def cop_and_resize(self, boxes: torch.Tensor, mask_size: int) -> torch.Tensor:
        assert len(boxes) == len(self), f"{len(boxes)} != {len(self)}"

        device = self.tensor.device
        batch_indexes = torch.arange(len(boxes), device=device).to(dtype=boxes.dtype)[:, None]
        rois = torch.cat([batch_indexes, boxes], dim=1)
        bit_masks = self.tensor.to(dtype=torch.float32)
        rois = rois.to(device=device)

        output = (
            ROIAlign((mask_size, mask_size), 1.0, 0, aligned=True)
            .forward(bit_masks[:, None, :, :], rois)
            .squeeze(1)
        )
        output = output >= 0.5

        return output

    def get_bounding_boxes(self) -> Boxes:
        boxes = torch.zeros(self.tensor.shape[0], 4, dtype=torch.float32)
        x_any = torch.any(self.tensor, dim=1)
        y_any = torch.any(self.tensor, dim=2)

        for index in range(self.tensor.shape[0]):
            x = torch.where(x_any[index, :])[0]
            y = torch.where(y_any[index, :])[0]

            if len(x) > 0 and len(y) > 0:
                boxes[index, :] = torch.as_tensor([x[0], y[0], x[-1] + 1, y[-1] + 1], dtype=torch.float32)

        return Boxes(boxes)

    @staticmethod
    def cat(bitmasks_list: List["BitMasks"]) -> "BitMasks":
        assert isinstance(bitmasks_list, (list, tuple))
        assert len(bitmasks_list) > 0
        assert all(isinstance(bitmask, BitMasks) for bitmask in bitmasks_list)

        cat_bitmasks = type(bitmasks_list[0])(torch.cat([bitmask.tensor for bitmask in bitmasks_list], dim=0))

        return cat_bitmasks


class PolygonMasks:
    def __init__(self, polygons: List[List[Union[torch.Tensor, np.ndarray]]]):
        assert isinstance(polygons, list), (
            f"Cannot create PolygonMasks: Expect a list of list of polygons per image. Got '{type(polygons)}' instead."
        )

        def make_array(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.cpu().numpy()

            return np.asarray(tensor).astype("float64")

        def process_polygons(polygons_per_instance: List[Union[torch.Tensor, np.ndarray]]) -> List[np.ndarray]:
            assert isinstance(polygons_per_instance, list), (
                f"Cannot create polygons: Expect a list of polygons per instance. "
                f"Got '{type(polygons_per_instance)}' instead."
            )

            polygons_per_instance = [make_array(polygon) for polygon in polygons_per_instance]

            for polygon in polygons_per_instance:
                assert len(polygon) % 2 == 0 and len(polygon) >= 6

            return polygons_per_instance

        self.polygons: List[List[np.ndarray]] = [
            process_polygons(polygons_per_instance) for polygons_per_instance in polygons
        ]

    def to(self) -> "PolygonMasks":
        return self

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def get_bounding_boxes(self) -> Boxes:
        boxes = torch.zeros(len(self.polygons), 4, dtype=torch.float32)

        for index, polygons_per_instance in enumerate(self.polygons):
            min_xy = torch.as_tensor([float("inf"), float("inf")], dtype=torch.float32)
            max_xy = torch.zeros(2, dtype=torch.float32)

            for polygon in polygons_per_instance:
                coords = torch.from_numpy(polygon).view(-1, 2).to(dtype=torch.float32)
                min_xy = torch.min(min_xy, torch.min(coords, dim=0).values)
                max_xy = torch.max(max_xy, torch.max(coords, dim=0).values)

            boxes[index, :2] = min_xy
            boxes[index, 2:] = max_xy

        return Boxes(boxes)

    def nonempty(self) -> torch.Tensor:
        keep = [1 if len(polygon) > 0 else 0 for polygon in self.polygons]

        return torch.from_numpy(np.asarray(keep, dtype=np.bool))

    def __getitem__(self, item: Union[int, slice, list[int], torch.BoolTensor]) -> "PolygonMasks":
        if isinstance(item, int):
            selected_polygons = [self.polygons[item]]
        elif isinstance(item, slice):
            selected_polygons = self.polygons[item]
        elif isinstance(item, list):
            selected_polygons = [self.polygons[index] for index in item]
        elif isinstance(item, torch.Tensor):
            if item.dtype == torch.bool:
                assert item.dim() == 1, item.shape

                item = item.nonzero().squeeze(1).cpu().numpy().tolist()
            elif item.dtype in [torch.int32, torch.int64]:
                item = item.cpu().numpy().tolist()
            else:
                raise ValueError(f"Unsupported tensor dtype{item.dtype} for indexing!")

            selected_polygons = [self.polygons[index] for index in item]
        else:
            raise TypeError(f"Unsupported type is used : type {type(item)}!")

        return PolygonMasks(selected_polygons)

    def __iter__(self) -> Iterator[List[np.ndarray]]:
        return iter(self.polygons)

    def __repr__(self) -> str:
        string = f"num_instances={len(self.polygons)}"

        return f"{self.__class__.__name__}({string})"

    def __len__(self) -> int:
        return len(self.polygons)

    def crop_and_resize(self, boxes: torch.Tensor, mask_size: int) -> torch.Tensor:
        assert len(boxes) == len(self), f"{len(boxes)} != {len(self)}"

        device = boxes.device
        boxes = boxes.to(torch.device("cpu"))

        results = [
            rasterize_polygons_within_box(polygon, box.numpy(), mask_size)
            for polygon, box in zip(self.polygons, boxes)
        ]

        if len(results) == 0:
            return torch.empty(0, mask_size, mask_size, dtype=torch.bool, device=device)

        return torch.stack(results, dim=0).to(device=device)

    def area(self):
        area = []

        for polygons_per_instance in self.polygons:
            area_per_instance = 0

            for polygon in polygons_per_instance:
                area_per_instance += polygon_area(polygon[0::2], polygon[1::2])

            area.append(area_per_instance)

        return torch.tensor(area)

    @staticmethod
    def cat(polygon_masks_list: List["PolygonMasks"]) -> "PolygonMasks":
        assert isinstance(polygon_masks_list, (list, tuple))
        assert len(polygon_masks_list) > 0
        assert all(isinstance(polygon_mask, PolygonMasks) for polygon_mask in polygon_masks_list)

        cat_polygon_masks = type(polygon_masks_list[0])(
            list(itertools.chain.from_iterable(polygon_mask.polygons for polygon_mask in polygon_masks_list))
        )

        return cat_polygon_masks


def polygon_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def polygon_to_bitmask(polygons: List[np.ndarray], height: int, width: int) -> np.ndarray:
    assert len(polygons) > 0, "COCO API does not support empty polygons"

    encodings = mask_util.frPyObjects(polygons, height, width)  # run length encoding (rle)
    encoding = mask_util.merge(encodings)

    return mask_util.decode(encoding).astype(np.bool)


def rasterize_polygons_within_box(polygons: List[np.ndarray], box: np.ndarray, mask_size: int) -> torch.Tensor:
    width, height = box[2] - box[0], box[3] - box[1]
    polygons = copy.deepcopy(polygons)

    for polygon in polygons:
        polygon[0::2] = polygon[0::2] - box[0]
        polygon[1::2] = polygon[1::2] - box[1]

    ratio_height = mask_size / max(float(height), 0.1)
    ratio_width = mask_size / max(float(width), 0.1)

    if ratio_height == ratio_width:
        for polygon in polygons:
            polygon *= ratio_height
    else:
        for polygon in polygons:
            polygon[0::2] *= ratio_width
            polygon[1::2] *= ratio_height

    mask = polygon_to_bitmask(polygons, mask_size, mask_size)
    mask = torch.from_numpy(mask)

    return mask
