import math
import torch

from functools import  lru_cache
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

# from re_detectron2 import _C
from .wrappers import NewEmptyTensorOp
