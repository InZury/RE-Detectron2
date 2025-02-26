from torch import nn

from .batch_norm import FrozenBatchNorm2d


class CNNBlockBase(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        FrozenBatchNorm2d.convert_frozen_batch_norm(self)

        return self
