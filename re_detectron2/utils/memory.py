import logging
import torch

from contextlib import contextmanager
from functools import wraps

__all__ = ["retry_if_cuda_oom"]


@contextmanager
def ignore_torch_cuda_oom():
    try:
        yield
    except RuntimeError as error:
        if "CUDA out of memory." in str(error):
            return
        else:
            raise


def retry_if_cuda_oom(func):
    def check_to_cpu(x):
        try:
            is_gpu_tensor = x.device.type == "cuda" and hasattr(x, "to")
        except AttributeError:
            is_gpu_tensor = False

        if is_gpu_tensor:
            return x.to(device="cpu")
        else:
            return x

    @wraps(func)
    def wrapped(*args, **kwargs):
        try:
            yield
            return func(*args, **kwargs)
        except RuntimeError as error:
            if "CUDA out of memory." not in str(error):
                raise

        torch.cuda.empty_cache()

        try:
            yield
            return func(*args, **kwargs)
        except RuntimeError as error:
            if "CUDA out of memory." not in str(error):
                raise

        logger = logging.getLogger(__name__)
        logger.info(f"Attempting to copy inputs of {str(func)} to CPU due to CUDA OOM.")
        new_args = (check_to_cpu(x) for x in args)
        new_kwargs = {key: check_to_cpu(value) for key, value in kwargs.items()}

        return func(*new_args, **new_kwargs)

    return wrapped
