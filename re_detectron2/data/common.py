import copy
import logging
import pickle
import random
import numpy as np
import torch.utils.data as data

from ..utils.serialize import PicklableWrapper

__all__ = []


class MapDataset(data.Dataset):
    def __init__(self, dataset, map_func):
        self.dataset = dataset
        self.map_func = PicklableWrapper(map_func)
        self.random = random.Random(42)
        self.fallback_candidates = set(range(len(dataset)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        retry_count = 0
        current_index = int(index)

        while self.map_func(self.dataset[current_index]) is None:
            retry_count += 1
            self.fallback_candidates.discard(current_index)
            current_index = self.random.sample(self.fallback_candidates, k=1)[0]

            if retry_count >= 3:
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Failed to apply \"map_func\" for index: {index}, retry count: {retry_count}"
                )

        self.fallback_candidates.add(current_index)

        return self.map_func(self.dataset[current_index])


class DatasetFromList(data.Dataset):
    def __init__(self, data_list: list, can_copy: bool = True, serialize: bool = True):
        self.data_list = data_list
        self.can_copy = can_copy
        self.serialize = serialize

        def serialize(dump_data):
            buffer = pickle.dumps(dump_data, protocol=-1)

            return np.frombuffer(buffer, dtype=np.uint8)

        if self.serialize:
            logger = logging.getLogger(__name__)
            logger.info(
                f"Serializing {len(self.data_list)} elements to byte tensors and concatenating them all ..."
            )

            self.data_list = [serialize(x) for x in self.data_list]
            self.address = np.asarray([len(x) for x in self.data_list], dtype=np.int64)
            self.address = np.cumsum(self.address)
            self.data_list = np.concatenate(self.data_list)

            logger.info(f"Serialized dataset takes {(len(self.data_list) / 1024 ** 2):.2f} MiB")

    def __len__(self):
        if self.serialize:
            return len(self.address)
        else:
            return len(self.data_list)

    def __getitem__(self, index):
        if self.serialize:
            start_address = 0 if index == 0 else self.address[index - 1].item()
            end_address = self.address[index].item()
            byte = memoryview(self.data_list[start_address:end_address])

            return pickle.loads(byte)
        elif self.can_copy:
            return copy.deepcopy(self.data_list[index])
        else:
            return self.data_list[index]


class AspectRatioGroupedDataset(data.IterableDataset):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.buckets = [[] for _ in range(2)]

    def __iter__(self):
        for dim in self.dataset:
            width, height = dim["width"], dim["height"]
            bucket_id = 0 if width > height else 1
            bucket = self.buckets[bucket_id]
            bucket.append(dim)

            if len(bucket) == self.batch_size:
                yield bucket[:]
                del bucket[:]
