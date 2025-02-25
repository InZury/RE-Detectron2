import itertools
import copy
import logging
import operator
import pickle
import torch.utils.data
import numpy as np

from fvcore.common.file_io import PathManager
from tabulate import tabulate
from termcolor import colored

# from ..structures import BoxMode
from ..utils.communication import get_world_size
from ..utils.environment import set_seed
from ..utils.logger import log_first_n
from .catalog import data_catalog, metadata_catalog
from .common import AspectRatioGroupedDataset, DatasetFromList, MapDataset
# from .dataset_mapper import DatasetMapper
# from .detection_utils import check_metadata_consistency
# from .samplers import InferenceSampler, RepeatFactorTrainingSampler, TrainingSampler
