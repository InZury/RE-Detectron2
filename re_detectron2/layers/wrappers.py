import torch
import torch.nn.functional as func

from typing import List, Any

ConvTranspose2d = torch.nn.ConvTranspose2d
BatchNorm2d = torch.nn.BatchNorm2d
Linear = torch.nn.Linear
interpolate = torch.nn.functional.interpolate


class NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape

        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, *grad_outputs) -> Any:
        shape = ctx.shape

        return NewEmptyTensorOp.apply(*grad_outputs, shape), None


class Conv2d(torch.nn.Conv2d):
    def __init__(self, *args, **kwargs):
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        if not torch.jit.is_scripting():
            if x.numel() == 0:
                if self.training:
                    assert not isinstance(
                        self.norm, torch.nn.SyncBatchNorm
                    ), "SyncBatchNorm does not support empty inputs!"

        x = func.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)

        return x


def cat(tensors: List[torch.Tensor], dim: int = 0):
    assert isinstance(tensors, (list, tuple))

    if len(tensors) == 1:
        return tensors[0]

    return torch.cat(tensors, dim)


def nonzero_tuple(x):
    if x.dim() == 0:
        return x.unsqueeze(0).nonzero().unbind(1)

    return x.nonzero().unbind(1)
