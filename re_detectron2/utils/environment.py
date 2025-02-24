import importlib
import importlib.util
import os
import sys
import random
import types
import torch
import numpy as np

from datetime import datetime

__all__ = ["set_seed", "setup_environment"]

TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])
ENVIRONMENT_SETUP_DONE = False


def set_seed(seed=None):
    if seed is None:
        seed = (
            os.getpid()
            + int(datetime.now().strftime("%S%f"))
            + int.from_bytes(os.urandom(2), "big")
        )
    np.random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    random.seed(seed)


def import_file(module_name, file_path, can_importable=False):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if can_importable:
        sys.modules[module_name] = module

    return module


def configure_libraries():
    disable_cv2 = int(os.environ.get("RE-DETECTRON2_DISABLE_CV2", False))

    if disable_cv2:
        sys.modules["cv2"] = types.ModuleType("NoneModule")
    else:
        os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"

        try:
            import cv2

            if int(cv2.__version__.split(".")[0]) >= 3:
                cv2.ocl.setUseOpenCL(False)
        except ModuleNotFoundError:
            pass

    def get_version(module, digit=2):
        return tuple(map(int, module.__version__.split(".")[:digit]))

    # fmt: off
    assert get_version(torch) >= (2, 6), "Requires torch >= 2.6"
    import fvcore
    assert get_version(fvcore, 3) >= (0, 1, 5), "Requires fvcore >= 0.1.5"
    import yaml
    assert get_version(yaml) >= (6, 0), "Requires yaml >= 6.0"
    # fmt: on


def setup_environment():
    global ENVIRONMENT_SETUP_DONE

    if ENVIRONMENT_SETUP_DONE:
        return

    ENVIRONMENT_SETUP_DONE = True

    configure_libraries()
    custom_module_path = os.environ.get("RE_DETECTRON2_ENV_MODULE")

    if custom_module_path:
        setup_custom_environment(custom_module_path)
    else:
        pass


def setup_custom_environment(custom_module):
    if custom_module.endswith(".py"):
        module = import_file("re_detectron2.utils.environments.custom_module", custom_module)
    else:
        module = importlib.import_module(custom_module)

    assert hasattr(module, "setup_environment") and callable(module.setup_environment), (
        f"Custom environment module defined in {custom_module} does not have "
        f"the required callable attribute \"setup_environment\"."
    )

    module.setup_environment()
