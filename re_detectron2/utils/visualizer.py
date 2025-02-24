import colorsys
import logging
import math
import cv2
import torch
import numpy as np
import pycocotools.mask as mask_util

from enum import Enum, unique
from fvcore.common.file_io import PathManager
from PIL import Image
from matplotlib import patches, lines
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

# from ..data import MetadataCatalog
# from ..structures import BitMask, Boxes, BoxMode, KeyPoints, PolygonMasks, RotatedBoxes
from .color_map import get_random_color
