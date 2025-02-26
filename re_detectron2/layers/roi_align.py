from torch import nn
from torchvision.ops import roi_align

roi_align = roi_align


class ROIAlign(nn.Module):
    def __init__(self, output_size, spatial_scale, sampling_ratio, aligned=True):
        super(ROIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned

    def forward(self, inputs, rois):
        assert rois.dim() == 2 and rois.size(1) == 5

        return roi_align(inputs, rois, self.output_size, self.spatial_scale, self.sampling_ratio, self.aligned)

    def __repr__(self):
        string = (f"output_size={self.output_size}, "
                  f"spatial_size={self.spatial_scale}, "
                  f"sampling_ratio={self.sampling_ratio}, "
                  f"aligned={self.aligned}")

        return f"{self.__class__.__name__}({string})"
