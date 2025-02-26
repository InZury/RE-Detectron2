import logging
import torch
import torch.distributed as dist

from torch import nn
from torch.autograd.function import Function
from torch.nn import functional as func

from ..utils import communication
from .wrappers import BatchNorm2d


class FrozenBatchNorm2d(nn.Module):
    _version = 3  # options

    def __init__(self, num_features, epsilon=1e-5):
        super().__init__()
        self.num_features = num_features
        self.epsilon = epsilon
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - epsilon)

    def forward(self, x):
        if x.reqires_grad:
            scale = self.weight * (self.running_var + self.epsilon).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)

            return x * scale + bias
        else:
            return func.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.epsilon
            )

    def load_from_stage_dict(
            self, state_dict, prefix, local_metadata, strict, missing_key, unexpected_keys, error_message
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            if prefix + "running_mean" not in state_dict:
                state_dict[prefix + "running_mean"] = torch.zeros_like(self.runnig_mean)
            if prefix + "running_var" not in state_dict:
                state_dict[prefix + "running_var"] = torch.ones_like(self.running_var)
        if version is not None and version < 3:
            logger = logging.getLogger(__name__)
            logger.info(f"FrozenBatchNorm {prefix.rstrip('.')} is upgraded to version 3.")

            state_dict[prefix + "running_var"] -= self.epsilon

        super().load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_key, unexpected_keys, error_message
        )

    def __repr__(self):
        return f"FrozenBatchNorm2d(num_features={self.num_features}, eps={self.epsilon}"

    @classmethod
    def convert_frozen_batch_norm(cls, module):
        batch_norm_module = nn.modules.batchnorm
        batch_norm_module = (batch_norm_module.BatchNorm2d, batch_norm_module.SyncBatchNorm)
        result = module

        if isinstance(module, batch_norm_module):
            result = cls(module.num_features)

            if module.affine:
                result.weight.data = module.weight.data.clone().detach()
                result.bias.data = module.bias.data.clone().detach()

            result.running_mean.data = module.running_mean.data
            result.running_var.data = module.running_var.data
            result.eps = module.eps
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batch_norm(child)

                if new_child is not child:
                    result.add_module(name, new_child)

        return result


class AllReduce(Function):
    @staticmethod
    def forward(ctx, inputs):
        input_list = [torch.zeros_like(inputs) for _ in range(dist.get_world_size())]
        dist.all_gather(input_list, inputs, async_op=False)
        inputs = torch.stack(input_list, dim=0)

        return torch.sum(inputs, dim=0)

    @staticmethod
    def backward(ctx, *grad_outputs):
        dist.all_reduce(grad_outputs, async_op=False)

        return grad_outputs


class NaiveSyncBatchNorm(BatchNorm2d):
    def __init__(self, *args, stats_mode="", **kwargs):
        super().__init__(*args, **kwargs)
        assert stats_mode in ["", "N"]

        self.stats_mode = stats_mode

    def forward(self, inputs):
        if communication.get_world_size() == 1 or not self.training:
            return super().forward(inputs)

        batch, channel = inputs.shape[0], inputs.shape[1]
        mean = torch.mean(inputs, dim=[0, 2, 3])
        mean_sqrt = torch.mean(inputs * inputs, dim=[0, 2, 3])

        if self.stats_mode == "":
            assert batch > 0, "SyncBatchNorm(stats_mode=\"\") does not support zero batch size."
            vector = torch.cat([mean, mean_sqrt], dim=0)
            vector = AllReduce.apply(vector) * (1.0 / dist.get_world_size())
            mean, mean_sqrt = torch.split(vector, channel)
            momentum = self.momentum
        else:
            if batch == 0:
                vector = torch.zeros([2 * channel + 1], device=mean.device, dtype=mean.dtype)
                vector = vector + inputs.sum()
            else:
                vector = torch.cat(
                    [mean, mean_sqrt, torch.ones([1], device=mean.device, dtype=mean.dtype)], dim=0
                )

            vector = AllReduce.apply(vector * batch)
            total_batch = vector[-1].detach()
            momentum = total_batch.clamp(max=1) * self.momentum
            total_batch = torch.max(total_batch, torch.ones_like(total_batch))
            mean, mean_sqrt, _ = torch.split(vector / total_batch, channel)

        var = mean_sqrt - mean * mean
        inverse_std = torch.rsqrt(var + self.eps)
        scale = self.weight * inverse_std
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)

        self.running_mean += momentum * (mean.detach() - self.running_mean)
        self.running_var += momentum * (var.detach() - self.running_var)

        return inputs * scale + bias


def get_norm(norm, out_channels):
    if isinstance(norm, str):
        if len(norm) == 0:
            return None

        norm = {
            "BatchNorm": BatchNorm2d,
            "SyncBatchNorm": nn.SyncBatchNorm,
            "FrozenBatchNorm": FrozenBatchNorm2d,
            "GroupNorm": lambda channels: nn.GroupNorm(32, channels),
            "nnSyncBatchNorm": nn.SyncBatchNorm,
            "naiveSyncBatchNorm": NaiveSyncBatchNorm
        }[norm]

    return norm(out_channels)
