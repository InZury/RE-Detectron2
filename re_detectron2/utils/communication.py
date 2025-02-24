import functools
import pickle
import torch
import numpy as np
import torch.distributed as dist

LOCAL_PROCESS_GROUP = None


def get_world_size() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1

    return dist.get_world_size()


def get_rank() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 0

    return dist.get_rank()


def get_local_rank() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 0

    assert LOCAL_PROCESS_GROUP is not None

    return dist.get_rank(group=LOCAL_PROCESS_GROUP)


def get_local_size() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1

    return dist.get_world_size(group=LOCAL_PROCESS_GROUP)


def synchronize():
    if not dist.is_available() or not dist.is_initialized():
        return

    world_size = dist.get_world_size()

    if world_size == 1:
        return

    dist.barrier()


@functools.lru_cache()
def get_global_gloo_group():
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


def serialize_to_tensor(data, group):
    backend = dist.get_backend(group)

    assert backend in ["gloo", "nccl"]

    device = torch.device("cpu" if backend == "gloo" else "cuda")
    buffer = pickle.dumps(data)

    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)

    return tensor


def pad_to_largest_tensor(tensor, group):
    world_size = dist.get_world_size(group=group)

    assert world_size >= 1, (
        "communication.gather/all_gather must be called from ranks within the given group!"
    )

    local_size = torch.tensor([tensor.numel()], dtype=torch.int64, device=tensor.device)
    size_list = [torch.zeros([1], dtype=torch.int64, device=tensor.device) for _ in range(world_size)]

    dist.all_gather(size_list, local_size, group=group)

    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    if local_size != max_size:
        padding = torch.zeros((max_size - local_size), dtype=torch.uint8, device=tensor.device)
        tensor = torch.cat((tensor, padding), dim=0)

    return size_list, tensor


def pick_all_gather(data, group=None):
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = get_global_gloo_group()
    if dist.get_world_size(group) == 1:
        return [data]

    tensor = serialize_to_tensor(data, group)
    size_list, tensor = pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)
    tensor_list = [torch.empty((max_size, ), dtype=torch.uint8, device=tensor.device) for _ in size_list]

    dist.all_gather(tensor_list, tensor, group=group)

    data_list = []

    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def pick_gather(data, target=0, group=None):
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = get_global_gloo_group()
    if dist.get_rank(group=group) == 1:
        return [data]

    rank = dist.get_rank(group=group)
    tensor = serialize_to_tensor(data, group)
    size_list, tensor = pad_to_largest_tensor(tensor, group)

    if rank == target:
        max_size = max(size_list)
        tensor_list = [torch.empty((max_size, ), dtype=torch.uint8, device=tensor.device) for _ in size_list]

        dist.gather(tensor, tensor_list, dst=target, group=group)

        data_list = []

        for size, tensor in zip(size_list, tensor_list):
            buffer = tensor.cpu().numpy().tobytes()[:size]
            data_list.append(pickle.loads(buffer))

        return data_list
    else:
        dist.gather(tensor, [], dst=target, group=group)

        return []


def get_shared_random_seed():
    ints = np.random.randint(2 ** 31)
    all_ints = pick_all_gather(ints)

    return all_ints[0]


def reduce_dictionary(inputs, average=True):
    world_size = get_world_size()

    if world_size < 2:
        return inputs

    with torch.no_grad():
        names = []
        values = []

        for key in sorted(inputs.keys()):
            names.append(key)
            values.append(inputs[key])

        values = torch.stack(values, dim=0)

        dist.reduce(values, dst=0)

        if dist.get_rank() == 0 and average:
            values /= world_size

        reduced_dict = {key: value for key, value in zip(names, values)}

    return reduced_dict
