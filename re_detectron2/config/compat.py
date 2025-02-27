import logging

from typing import List, Optional, Tuple, Union

from .config import ConfigNode
from .defaults import CONFIG

__all__ = ["upgrade_config", "downgrade_config"]


class RenameConverter:
    RENAME: List[Tuple[str, str]] = []

    @classmethod
    def upgrade(cls, config: ConfigNode) -> None:
        for old, new in cls.RENAME:
            rename(config, old, new)

    @classmethod
    def downgrade(cls, config: ConfigNode) -> None:
        for old, new in cls.RENAME[::-1]:
            rename(config, new, old)


class ConverterV1(RenameConverter):
    RENAME = [("MODEL.RPN_HEAD.NAME", "MODEL.RPN.HEAD_NAME")]


class ConverterV2(RenameConverter):
    RENAME = [
        ("MODEL.WEIGHT", "MODEL.WEIGHTS"),
        ("MODEL.PANOPTIC_FPN.SEMANTIC_LOSS_SCALE", "MODEL.SEMANTIC_SEGMENTATION_HEAD.LOSS_WEIGHT"),
        ("MODEL.PANOPTIC_FPN.RPN_LOSS_SCALE", "MODEL.RPN.LOSS_WEIGHT"),
        ("MODEL.PANOPTIC_FPN.INSTANCE_LOSS_SCALE", "MODEL.PANOPTIC_FPN.INSTANCE_LOSS_WEIGHT"),
        ("MODEL.PANOPTIC_FPN.COMBINE_ON", "MODEL.PANOPTIC_FPN.COMBINE.ENABLED"),
        (
            "MODEL.PANOPTIC_FPN.COMBINE_OVERLAP_THRESHOLD",
            "MODEL.PANOPTIC_FPN.COMBINE.OVERLAP_THRESH",
        ),
        (
            "MODEL.PANOPTIC_FPN.COMBINE_STUFF_AREA_LIMIT",
            "MODEL.PANOPTIC_FPN.COMBINE.STUFF_AREA_LIMIT",
        ),
        (
            "MODEL.PANOPTIC_FPN.COMBINE_INSTANCES_CONFIDENCE_THRESHOLD",
            "MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH",
        ),
        ("MODEL.ROI_HEADS.SCORE_THRESH", "MODEL.ROI_HEADS.SCORE_THRESH_TEST"),
        ("MODEL.ROI_HEADS.NMS", "MODEL.ROI_HEADS.NMS_THRESH_TEST"),
        ("MODEL.RETINANET.INFERENCE_SCORE_THRESHOLD", "MODEL.RETINANET.SCORE_THRESH_TEST"),
        ("MODEL.RETINANET.INFERENCE_TOPK_CANDIDATES", "MODEL.RETINANET.TOP_K_CANDIDATES_TEST"),
        ("MODEL.RETINANET.INFERENCE_NMS_THRESHOLD", "MODEL.RETINANET.NMS_THRESH_TEST"),
        ("TEST.DETECTIONS_PER_IMG", "TEST.DETECTIONS_PER_IMAGE"),
        ("TEST.AUG_ON", "TEST.AUG.ENABLED"),
        ("TEST.AUG_MIN_SIZES", "TEST.AUG.MIN_SIZES"),
        ("TEST.AUG_MAX_SIZE", "TEST.AUG.MAX_SIZE"),
        ("TEST.AUG_FLIP", "TEST.AUG.FLIP"),
    ]

    @classmethod
    def upgrade(cls, config: ConfigNode) -> None:
        super().upgrade(config)

        if config.MODEL.META_ARCHITECTURE == "RetinaNet":
            rename(
                config,
                "MODEL.RETINANET.ANCHOR_ASPECT_RATIOS",
                "MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS"
            )
            rename(
                config,
                "MODEL.RETINANET.ANCHOR_SIZES",
                "MODEL.ANCHOR_GENERATOR.SIZES"
            )

            del config["MODEL"]["RPN"]["ANCHOR_SIZES"]
            del config["MODEL"]["RPN"]["ANCHOR_ASPECT_RATIOS"]
        else:
            rename(
                config,
                "MODEL.RPN.ANCHOR_ASPECT_RATIOS",
                "MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS"
            )
            rename(
                config,
                "MODEL.PRN.ANCHOR_SIZES",
                "MODEL.ANCHOR_GENERATOR.SIZES"
            )

            del config["MODEL"]["RETINANET"]["ANCHOR_SIZES"]
            del config["MODEL"]["RETINANET"]["ANCHOR_ASPECT_RATIOS"]

        del config["MODEL"]["RETINANET"]["ANCHOR_STRIDES"]

    @classmethod
    def downgrade(cls, config: ConfigNode) -> None:
        super().downgrade(config)

        rename(
            config,
            "MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS",
            "MODEL.RPN.ANCHOR_ASPECT_RATIOS"
        )
        rename(
            config,
            "MODEL.ANCHOR_GENERATOR.SIZES",
            "MODEL.PRN.ANCHOR_SIZES"
        )

        config.MODEL.RETINANET.ANCHOR_ASPECT_RATIOS = config.MODEL.RPN.ANCHOR_ASPECT_RATIOS
        config.MODEL.RETINANET.ANCHOR_SIZES = config.MODEL.RPN.ANCHOR_SIZES
        config.MODEL.RETINANET.ANCHOR_STRIDES = []


def upgrade_config(config: ConfigNode, to_version: Optional[int] = None) -> ConfigNode:
    config = config.clone()

    if to_version is None:
        to_version = CONFIG.VERSION

    assert config.VERSION <= to_version, f"Cannot upgrade from v{config.VERSION} to v{to_version}!"

    for key in range(config.VERSION, to_version):
        converter = globals()[f"ConverterV{key + 1}"]
        converter.upgrade(config)
        config.VERSION = key + 1

    return config


def downgrade_config(config: ConfigNode, to_version: int) -> ConfigNode:
    config = config.clone()

    assert config.VERSION >= to_version, f"Cannot downgrade from v{config.VERSION} to v{to_version}!"

    for key in range(config.VERSION, to_version, -1):
        converter = globals()[f"ConverterV{key}"]
        converter.downgrade(config)
        config.VERSION = key - 1

    return config


def guess_version(config: ConfigNode, filename: str) -> int:
    logger = logging.getLogger(__name__)

    def _has(name: str) -> bool:
        current = config

        for num in name.split('.'):
            if num not in current:
                return False

            current = current[num]

        return True

    if _has("MODEL_WEIGHT") or _has("TEST.AUG_ON"):
        result = 1
        logger.warning(f"Config \"{filename}\" has no VERSION. Assuming it to be v{result}")
    else:
        result = CONFIG.VERSION
        logger.warning(f"Config \"{filename}\" has no VERSION. Assuming it to be compatible with latest v{result}")

    return result


def rename(config: ConfigNode, old: str, new: str) -> None:
    old_keys = old.split('.')
    new_keys = new.split('.')

    def set_key(key_list: List[str], val: str) -> None:
        current = config

        for key in key_list[:-1]:
            if key not in current:
                current[key] = ConfigNode()

            current = current[key]

        current[key_list[-1]] = val

    def get_key(key_list: List[str]) -> Union[ConfigNode, str]:
        current = config

        for key in key_list:
            current = current[key]

        return current

    def del_key(key_list: List[str]) -> None:
        current = config

        for key in key_list[:-1]:
            current = current[key]

        del current[key_list[-1]]

        if len(current) == 0 and len(key_list) > 1:
            del_key(key_list[:-1])

    set_key(new_keys, get_key(old_keys))
    del_key(old_keys)
