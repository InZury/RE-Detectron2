import torch
import fvcore.nn.weight_init as weight_init

from torch import nn
from torch.nn import functional as func

from .batch_norm import get_norm
from .wrappers import Conv2d


class ASPP(nn.Module):  # Atrous Spatial Pyramid Pooling (ASPP)
    def __init__(self, in_channels, out_channels, dilations, norm, activation, pool_kernel_size=None, dropout=0.0):
        super(ASPP, self).__init__()
        assert len(dilations) == 3, f"ASPP expects 3 dilations, got {len(dilations)}."
        use_bias = norm == ""

        self.pool_kernel_size = pool_kernel_size
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(
            Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=use_bias,
                norm=get_norm(norm, out_channels),
                activation=activation
            )
        )

        weight_init.c2_xavier_fill(self.convs[-1])

        # Atrous convolutions
        for dilation in dilations:
            self.convs.append(
                Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                    bias=use_bias,
                    norm=get_norm(norm, out_channels),
                    activation=activation
                )
            )

            weight_init.c2_xavier_fill(self.convs[-1])

        if pool_kernel_size is None:
            image_pooling = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Conv2d(in_channels, out_channels, 1, bias=True, activation=activation)
            )
        else:
            image_pooling = nn.Sequential(
                nn.AvgPool2d(kernel_size=pool_kernel_size, stride=1),
                Conv2d(in_channels, out_channels, 1, bias=True, activation=activation)
            )

        weight_init.c2_xavier_fill(image_pooling[1])

        self.convs.append(image_pooling)
        self.project = Conv2d(
            5 * out_channels,
            out_channels,
            kernel_size=1,
            bias=use_bias,
            norm=get_norm(norm, out_channels),
            activation=activation
        )

        weight_init.c2_xavier_fill(self.project)

    def forward(self, x):
        size = x.shape[-2:]

        if self.pool_kernel_size is not None:
            if size[0] % self.pool_kernel_size[0] or size[1] % self.pool_kernel_size[1]:
                raise ValueError(
                    f"\"pool_kernel_size\" must be divisible by the shape of inputs."
                    f"Input size: {size} \"pool_kernel_size\": {self.pool_kernel_size}"
                )

        result = []

        for conv in self.convs:
            result.append(conv(x))

        result[-1] = func.interpolate(result[-1], size=size, mode="bilinear", align_corners=False)
        result = torch.cat(result, dim=1)
        result = self.project(result)
        result = func.dropout(result, self.dropout, training=self.training) if self.dropout > 0 else result

        return result
