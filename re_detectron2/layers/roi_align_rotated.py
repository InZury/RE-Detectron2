from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from re_detectron2 import CFile


class ROIAlignRotatedFunc(Function):
    @staticmethod
    def forward(ctx, inputs, roi, output_size, spatial_scale, sampling_ratio):
        ctx.save_for_backward(roi)
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        ctx.input_shape = inputs.size()

        output = CFile.roi_align_rotated_forward(
            inputs, roi, spatial_scale, output_size[0], output_size[1], sampling_ratio
        )

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (rois, ) = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        sampling_ratio = ctx.sampling_ratio
        batch_size, channel, height, width = ctx.input_shape
        grad_input = CFile.roi_align_rotated_forward(
            grad_output,
            rois,
            spatial_scale,
            output_size[0],
            output_size[1],
            batch_size,
            channel,
            height,
            width,
            sampling_ratio
        )

        return grad_input, None, None, None, None, None


roi_align_rotated = ROIAlignRotatedFunc.apply


class ROIAlignRotated(nn.Module):
    def __init__(self, output_size, spatial_scale, sampling_ratio):
        super(ROIAlignRotated, self).__init__()
        self.output_size = output_size,
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def forward(self, inputs, rois):
        assert rois.dim() == 2 and rois(1) == 6

        return roi_align_rotated(
            inputs, rois, self.output_size, self.spatial_scale, self.sampling_ratio
        )

    def __repr__(self):
        string = (f"output_size={self.output_size}, "
                  f"spatial_scale={self.spatial_scale}, "
                  f"sampling_ratio={self.sampling_ratio}")

        return f"{self.__class__.__name__}({string})"
