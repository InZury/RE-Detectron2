import numpy as np
import torch

from torch.nn import functional as func

__all__ = []

BYTES_PER_FLOAT = 4

# Check to memory size
GPU_MEMORY_LIMIT = 1024 ** 3


def do_paste_mask(masks, boxes, image_height, image_width, skip_empty=True):
    device = masks.device

    if skip_empty:
        x0_int, y0_int = torch.clamp(boxes.min(dim=0).values.floor()[:2] - 1, min=0).to(dtype=torch.int32)
        x1_int = torch.clamp(boxes[:, 2].max().ceil() + 1, max=image_width).to(dtype=torch.int32)
        y1_int = torch.clamp(boxes[:, 2].max().ceil() + 1, max=image_height).to(dtype=torch.int32)
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = image_width, image_height

    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)
    num_mask = masks.shape[0]

    image_y = torch.arange(y0_int, y1_int, device=device, dtype=torch.float32) + 0.5
    image_x = torch.arange(x0_int, x1_int, device=device, dtype=torch.float32) + 0.5
    image_y = (image_y - y0) / (y1 - y0) * 2 - 1
    image_x = (image_x - x0) / (x1 - x0) * 2 - 1

    grid_x = image_x[:, None, :].expand(num_mask, image_y.size(1), image_x.size(1))
    grid_y = image_y[:, :, None].expand(num_mask, image_y.size(1), image_x.size(1))
    grid = torch.stack([grid_x, grid_y], dim=3)

    image_masks = func.grid_sample(masks.to(dtype=torch.float32), grid, align_corners=False)

    if skip_empty:
        return image_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return image_masks[:, 0], ()


def paste_masks_in_image(masks, boxes, image_shape, threshold=0.5):
    assert masks.shape[-1] == masks.shape[-2], "Only square mask predictions are supported"

    num_mask = len(masks)

    if num_mask == 0:
        return masks.new_empty((0, ) + image_shape, dtype=torch.uint8)
    if not isinstance(boxes, torch.Tensor):
        boxes = boxes.tensor

    device = boxes.device

    assert len(boxes) == num_mask, boxes.shape

    image_height, image_width = image_shape

    if device.type == "cpu":
        num_chunks = num_mask
    else:
        num_chunks = int(np.ceil(num_mask * int(image_height) * int(image_width) * BYTES_PER_FLOAT / GPU_MEMORY_LIMIT))

        assert num_chunks <= num_mask, "Default GPU_MEMORY_LIMIT in mask_operations.py is too small; try increasing it"

    chunks = torch.chunk(torch.arange(num_mask, device=device), num_chunks)

    image_masks = torch.zeros(
        num_mask, image_height, image_width, device=device, dtype=torch.bool if threshold >= 0 else torch.uint8
    )

    for indexes in chunks:
        masks_chunk, spatial_indexes = do_paste_mask(
            masks[indexes, None, :, :], boxes[indexes], image_height, image_width, skip_empty=device.type == "cpu"
        )

        if threshold >= 0:
            masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
        else:
            masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

        image_masks[(indexes, ) + spatial_indexes] = masks_chunk

    return image_masks
