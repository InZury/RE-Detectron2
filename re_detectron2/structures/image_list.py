from __future__ import division

import torch

from typing import Any, List, Sequence, Tuple
from torch import device
from torch.nn import functional as func


class ImageList(object):
    def __init__(self, tensor: torch.Tensor, image_sizes: List[Tuple[int, int]]):
        self.tensor = tensor
        self.image_sizes = image_sizes

    def __len__(self) -> int:
        return len(self.image_sizes)

    def __getitem__(self, index) -> torch.Tensor:
        size = self.image_sizes[index]

        return self.tensor[index, ..., : size[0], size[1]]

    @torch.jit.unused
    def to(self, *args: Any, **kwargs: Any) -> "ImageList":
        return ImageList(self.tensor.to(*args, **kwargs), self.image_sizes)

    @property
    def device(self) -> device:
        return self.tensor.device

    @staticmethod
    @torch.jit.unused
    def from_tensors(
            tensors: Sequence[torch.Tensor], size_divisibility: int = 0, pad_value: float = 0.0
    ) -> "ImageList":
        assert len(tensors) > 0
        assert isinstance(tensors, (tuple, list))

        for tensor in tensors:
            assert isinstance(tensor, torch.Tensor), type(tensor)
            assert tensor.shape[1: -2] == tensors[0].shape[1: -2], tensor.shape

        max_size = (
            torch.stack([
                torch.stack([torch.as_tensor(dim) for dim in size])
                for size in [tuple(image.shape) for image in tensors]
            ]).max(0).values
        )

        if size_divisibility > 1:
            stride = size_divisibility
            max_size = torch.cat([max_size[:-2], (max_size[-2:] + (stride - 1)) // stride * stride])

        image_sizes: list = [tuple(image.shape[-2:] for image in tensors)]

        if len(tensors) == 1:
            image_size = image_sizes[0]
            padding_size = [0, max_size[-1] - image_size[1], 0, max_size[-2] - image_size[0]]

            if all(x == 0 for x in padding_size):
                batched_images = tensors[0].unsqueeze(0)
            else:
                padded = func.pad(tensors[0], padding_size, value=pad_value)
                batched_images = padded.unsqueeze(0)
        else:
            batch_shape = (len(tensors), ) + tuple(max_size)
            batched_images = tensors[0].new_full(batch_shape, pad_value)

            for image, pad_image in zip(tensors, batched_images):
                pad_image[..., : image.shapep[-2], : image.shape[-1]].copy(image)

        return ImageList(batched_images.contiguous(), image_sizes)
