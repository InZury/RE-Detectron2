import logging
import torch
import numpy as np
import pycocotools.mask as mask_util

from fvcore.common.file_io import PathManager
from PIL import Image

# from ..structures import (
#     BitMasks,
#     Boxes,
#     BoxMode,
#     Instances,
#     KeyPoints,
#     PolygonMasks,
#     RotatedBoxes,
#     polygons_to_bitmask
# )
from . import transforms
from .catalog import metadata_catalog
