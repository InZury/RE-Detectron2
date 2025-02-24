import importlib
import importlib.util
import os
import re
import subprocess
import sys
import PIL
import torch
import torchvision
import numpy as np

from collections import defaultdict
from tabulate import tabulate

__all__ = []


def collect_torch_environment():
    try:
        import torch.__config__

        return torch.__config__.show()
    except ImportError:
        from torch.utils.collect_env import get_pretty_env_info

        return get_pretty_env_info()


def get_module_environment():
    module_name = "RE_DETECTRON2_ENV_MODULE"

    return module_name, os.environ.get(module_name, "<not set>")


def detect_compute_compatibility(cuda_home, so_file):
    try:
        cuobjdump_file = os.path.join(cuda_home, "bin", "cuobjdump")

        if os.path.isfile(cuobjdump_file):
            output = subprocess.check_output(f"\"{cuobjdump_file}\" -- list-elf \"{so_file}\"", shell=True)
            output = output.decode("utf-8").strip().split("\n")
            architecture = []

            for line in output:
                line = re.findall(r"\.sm_([0-9]*)\.", line)[0]
                architecture.append(line)

            architecture = sorted(set(architecture))

            return ", ".join(architecture)
        else:
            return f"{so_file}; cannot find cuobjdump"
    except FileExistsError:
        return so_file


def collect_environment_info():
    from torch.utils.cpp_extension import CUDA_HOME

    has_gpu = torch.cuda.is_available()
    torch_version = torch.__version__
    has_rocm = False
    rocm = ""

    if tuple(map(int, torch_version.split(".")[:2])) >= (1, 5):
        from torch.utils.cpp_extension import ROCM_HOME

        if (getattr(torch.version, "hip", None) is not None) and (ROCM_HOME is not None):
            has_rocm = True
            rocm = str(ROCM_HOME)

    has_cuda = has_gpu and (not has_rocm)

    data = [("sys.platform", sys.platform), ("Python", sys.version.replace("\n", "")), ("numpy", np.__version__)]

    try:
        import re_detectron2  # noqa

        data.append(("re_detectron2", f"{re_detectron2.__version__}@{os.path.dirname(re_detectron2.__file__)}"))
    except ImportError:
        data.append(("re_detectron2", "failed to import"))

    try:
        from re_detectron2 import CONFIG
    except ImportError:
        data.append(("re_detectron2.CONFIG", "failed to import. re_detectron2 is not built correctly"))

        if sys.platform != "win32":
            try:
                cxx = os.environ.get("CXX", "c++")
                cxx = subprocess.check_output(f"\"{cxx}\" --version", shell=True)
                cxx = cxx.decode("utf-8").strip().split("\n")[0]
            except subprocess.SubprocessError:
                cxx = "Not found"

            data.append(("Compiler", cxx))

            if has_cuda and CUDA_HOME is not None:
                try:
                    nvcc = os.path.join(CUDA_HOME, "bin", "nvcc")
                    nvcc = subprocess.check_output(f"\"{nvcc}\" -V", shell=True)
                    nvcc = nvcc.decode("utf-8").strip().split("\n")[-1]
                except subprocess.SubprocessError:
                    nvcc = "Not found"

                data.append(("CUDA compiler", nvcc))
        else:
            data.append(("Compiler", CONFIG.get_compiler_version()))
            data.append(("CUDA compiler", CONFIG.get_cuda_version()))

            if has_cuda:
                data.append(
                    ("re_detectron2 architecture flags", detect_compute_compatibility(CUDA_HOME, CONFIG.__file__))
                )

        data.append(get_module_environment())
        data.append(("Pytorch", f"{torch_version}@{os.path.dirname(torch.__file__)}"))
        data.append(("PyTorch debug build", torch.version.debug))
        data.append(("GPU available", has_gpu))

        if has_gpu:
            devices = defaultdict(list)

            for index in range(torch.cuda.device_count()):
                capability = ".".join((str(x) for x in torch.cuda.get_device_capability(index)))
                name = torch.cuda.get_device_name(index) + f" (arch={capability}"
                devices[name].append(str(index))

            for name, device in devices.items():
                data.append(("GPU" + ",".join(device), name))

            if has_rocm:
                message = " - invalid!" if not os.path.isdir(rocm) else ""
                data.append(("ROCM_HOME", rocm + message))
            else:
                message = " - invalid!" if not os.path.isdir(CUDA_HOME) else ""
                data.append(("CUDA_HOME", str(CUDA_HOME) + message))

                cuda_architecture_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)

                if cuda_architecture_list:
                    data.append(("TORCH_CUDA_ARCH_LIST", cuda_architecture_list))

        data.append(("Pillow", PIL.__version__))

        try:
            data.append(("torchvision", f"{str(torchvision.__version__)}@{os.path.dirname(torchvision.__file__)}"))

            if has_cuda:
                try:
                    torchvision_config = importlib.util.find_spec("torchvision._C").origin
                    message = detect_compute_compatibility(CUDA_HOME, torchvision_config)
                    data.append(("torchvision architecture flags", message))
                except ImportError:
                    data.append(("torchvision._C", "Not found"))
        except AttributeError:
            data.append(("torchvision", "unknown"))

        try:
            import fvcore

            data.append(("fvcore", fvcore.__version__))
        except ImportError:
            import fvcore
            pass

        try:
            import cv2

            data.append(("cv2", cv2.__version__))
        except ImportError:
            import cv2
            data.append(("cv2", "Not found"))

        environment = tabulate(data) + "\n"
        environment += collect_torch_environment()

        return environment


if __name__ == "__main__":
    try:
        import re_detectron2
    except ImportError:
        print(collect_environment_info())

    if torch.cuda.is_available():
        for k in range(torch.cuda.device_count()):
            cuda = f"cuda:{k}"

            try:
                tensor = torch.tensor([1, 2, 0], dtype=torch.float32)
                tensor = tensor.to(cuda)
            except ValueError:
                print(f"Unable to copy tensor to device={cuda}")
