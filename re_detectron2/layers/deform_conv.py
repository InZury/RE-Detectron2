import math
import torch

from functools import lru_cache
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from re_detectron2 import CFile
from .wrappers import NewEmptyTensorOp


class DeformConvFunc(Function):
    @staticmethod
    def forward(
        ctx,
        inputs,
        offset,
        weight,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deformable_groups=1,
        image2column_step=64
    ):
        assert inputs is not None, ValueError("Inputs cannot be None data!")

        if inputs.dim() != 4:
            raise ValueError(f"Expected 4D tensor as inputs, got {inputs.dim()}D tensor instead.")

        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.image2column_step = image2column_step
        ctx.save_for_backward(inputs, offset, weight)
        ctx.buffers = [inputs.new_empty(0), inputs.new_empty(0)]

        output = inputs.new_empty(
            DeformConvFunc.output_size(inputs, weight, ctx.padding, ctx.dilation, ctx.stride)
        )

        if not inputs.is_cuda:
            raise NotImplementedError("Deformable Conv is not supported on CPUs!")
        else:
            current_image2column_step = DeformConvFunc.compute_image2column_step(inputs.shape[0], ctx.image2column_step)
            assert inputs.shape[0] % current_image2column_step == 0, "image2column step must divide batch size."

            CFile.deform_conv_forward(
                inputs,
                weight,
                offset,
                output,
                ctx.buffers[0],
                ctx.buffers[1],
                weight.size(3),
                weight.size(2),
                ctx.stride[1],
                ctx.stride[0],
                ctx.padding[1],
                ctx.padding[0],
                ctx.dilation[1],
                ctx.dilation[0],
                ctx.groups,
                ctx.deformable_groups,
                current_image2column_step
            )

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        inputs, offset, weight = ctx.save_for_backward
        grad_input = grad_offset = grad_weight = None

        if not grad_output.is_cuda:
            raise NotImplementedError("Deformable Conv is not supported on CPUs!")
        else:
            current_image2column_step = DeformConvFunc.compute_image2column_step(inputs.shape[0], ctx.image2column_step)

            assert inputs.shape[0] % current_image2column_step == 0, "image2column step must divide batch size."

            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                grad_input = torch.zeros_like(inputs)
                grad_offset = torch.zeros_like(offset)

                CFile.deform_conv_backward_input(
                    inputs,
                    offset,
                    grad_output,
                    grad_input,
                    grad_offset,
                    weight,
                    ctx.buffers[0],
                    weight.size(3),
                    weight.size(2),
                    ctx.stride[1],
                    ctx.stride[0],
                    ctx.padding[1],
                    ctx.padding[0],
                    ctx.dilation[1],
                    ctx.dilation[0],
                    ctx.groups,
                    ctx.deformable_groups,
                    current_image2column_step
                )

            if ctx.needs_input_grad[2]:
                grad_weight = torch.zeros_like(weight)
                CFile.deform_conv_backward_filter(
                    inputs,
                    offset,
                    grad_output,
                    grad_weight,
                    ctx.buffers[0],
                    ctx.buffers[1],
                    weight.size(3),
                    weight.size(2),
                    ctx.stride[1],
                    ctx.stride[0],
                    ctx.padding[1],
                    ctx.padding[0],
                    ctx.dilation[1],
                    ctx.dilation[0],
                    ctx.groups,
                    ctx.deformable_groups,
                    1,
                    current_image2column_step
                )

        return grad_input, grad_offset, grad_weight, None, None, None, None, None, None

    @staticmethod
    def output_size(inputs, weight, padding, dilation, stride):
        channels = weight.size(0)
        output_size = (inputs.size(0), channels)

        for dim in range(inputs.dim() - 2):
            in_size = inputs.size(dim + 2)
            pad = padding[dim]
            kernel = dilation[dim] * (weight.size(dim + 2) - 1) + 1
            stride_dim = stride[dim]
            output_size += ((in_size + (2 * pad) - kernel) // stride_dim + 1, )

        if not all(map(lambda size: size > 0, output_size)):
            raise ValueError(
                f"convolution input is too small (output would be {'x'.join(map(str, output_size))})"
            )

        return output_size

    @staticmethod
    @lru_cache(maxsize=128)
    def compute_image2column_step(input_size, default_size):
        if input_size <= default_size:
            return input_size

        best_step = 1

        for step in range(2, min(int(math.sqrt(input_size)) + 1, default_size)):
            if input_size % step == 0:
                if input_size // step <= default_size:
                    return input_size // step

                best_step = step

        return best_step


class ModulatedDeformConvFunc(Function):
    @staticmethod
    def forward(
        ctx,
        inputs,
        offset,
        mask,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deformable_groups=1
    ):
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.with_bias = bias is not None

        if not ctx.with_bias:
            bias = inputs.new_empty(1)
        if not inputs.is_cuda:
            raise NotImplementedError("Deformable Conv is not supported on CPUs!")
        if weight.requires_grad or mask.requires_grad or offset.requires_grad or inputs.requires_grad:
            ctx.save_for_backward(inputs, offset, mask, weight, bias)

        output = inputs.new_empty(ModulatedDeformConvFunc.infer_shape(ctx, inputs, weight))
        ctx.buffers = [inputs.new_empty(0), inputs.new_empty(0)]

        CFile.modulated_deform_conv_forward(
            inputs,
            weight,
            bias,
            ctx.buffers[0],
            offset,
            mask,
            output,
            ctx.buffers[1],
            weight.shape[2],
            weight.shape[3],
            ctx.stride,
            ctx.stride,
            ctx.padding,
            ctx.padding,
            ctx.dilation,
            ctx.dilation,
            ctx.groups,
            ctx.deformable_groups,
            ctx.with_bias
        )

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError("Deformable Conv is not supported on CPUs!")

        inputs, offset, mask, weight, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(inputs)
        grad_offset = torch.zeros_like(offset)
        grad_mask = torch.zeros_like(mask)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)

        CFile.modulated_deform_conv_backward(
            input,
            weight,
            bias,
            ctx.buffers[0],
            offset,
            mask,
            ctx.buffers[1],
            grad_input,
            grad_weight,
            grad_bias,
            grad_offset,
            grad_mask,
            grad_output,
            weight.shape[2],
            weight.shape[3],
            ctx.stride,
            ctx.stride,
            ctx.padding,
            ctx.padding,
            ctx.dilation,
            ctx.dilation,
            ctx.groups,
            ctx.deformable_groups,
            ctx.with_bias
        )

        if not ctx.with_bias:
            grad_bias = None

        return grad_input, grad_offset, grad_mask, grad_weight, grad_bias, None, None, None, None, None

    @staticmethod
    def infer_shape(ctx, inputs, weight):
        context = inputs.size(0)
        channels_out = weight.size(0)
        height, width = inputs.shape[2:4]
        kernel_height, kernel_width = weight.shape[2:4]
        height_out = (height + 2 * ctx.padding - (ctx.dilation * kernel_height - 1) + 1) // ctx.stride + 1
        width_out = (width + 2 * ctx.padding - (ctx.dilation * kernel_width - 1) + 1) // ctx.stride + 1

        return context, channels_out, height_out, width_out


deform_conv = DeformConvFunc.apply
modulated_deform_conv = ModulatedDeformConvFunc.apply


class DeformConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deformable_groups=1,
        bias=False,
        norm=None,
        activation=None
    ):
        super(DeformConv, self).__init__()
        assert not bias
        assert in_channels % groups == 0, f"in_channels {in_channels} cannot be divisible by groups {groups}."
        assert out_channels % groups == 0, f"out_channels {out_channels} cannot be divisible by groups {groups}."

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.norm = norm
        self.activation = activation
        self.bias = None
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size))

        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")

    def forward(self, x, offset):
        if x.numel() == 0:
            output_shape = get_output(x, self.padding, self.dilation, self.kernel_size, self.stride)
            output_shape = [x.shape[0], self.weight.shape[0]] + output_shape

            return NewEmptyTensorOp.apply(x, output_shape)

        x = deform_conv(
            x,
            offset,
            self.weight,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.deformable_groups
        )

        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)

        return x

    def extra_repr(self):
        return (f"in_channels={self.in_channels}, "
                f"out_channels={self.out_channels}, "
                f"kernel_size={self.kernel_size}, "
                f"stride={self.stride}, "
                f"padding={self.padding}, "
                f"dilation={self.dilation}, "
                f"groups={self.groups}, "
                f"deformable_group={self.deformable_groups}, "
                f"bias=False")


class ModulatedDeformConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deformable_groups=1,
        bias=True,
        norm=None,
        activation=None
    ):
        super(ModulatedDeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias
        self.norm = norm
        self.activation = activation
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None

        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")

        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x, offset, mask):
        if x.numel() == 0:
            output_shape = get_output(x, self.padding, self.dilation, self.kernel_size, self.stride)
            output_shape = [x.shape[0], self.weight.shape[0]] + output_shape

            return NewEmptyTensorOp.apply(x, output_shape)

        x = modulated_deform_conv(
            x,
            offset,
            mask,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.deformable_groups
        )

        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)

        return x

    def extra_repr(self):
        return (f"in_channels={self.in_channels}, "
                f"out_channels={self.out_channels}, "
                f"kernel_size={self.kernel_size}, "
                f"stride={self.stride},"
                f"padding={self.padding},"
                f"dilation={self.dilation},"
                f"groups={self.groups},"
                f"deformable_groups={self.deformable_groups}, "
                f"bias={self.with_bias}")


def get_output(x, padding, dilation, kernel, stride):
    return [(inputs + 2 * padding - (dilation * (kernel - 1) + 1)) // stride + 1
            for inputs, padding, dilation, kernel, stride in zip(
                x.shape[-2:], padding, dilation, kernel, stride
        )
    ]
