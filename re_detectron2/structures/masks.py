import copy
import itertools
import torch
import numpy as np
import pycocotools.mask as mask_util

from typing import Any, Iterator, List, Union

# from ..layers.roi_align import ROIAlign
from .boxes import Boxes
