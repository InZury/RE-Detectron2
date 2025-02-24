import itertools
import torch

from typing import Any, Dict, List, Tuple, Union


class Instances:
    def __init__(self, image_size: Tuple[int, int], **kwargs: Any):
        self.image_size = image_size
        self.fields: Dict[str, Any] = {}

        for key, value in kwargs.items():
            self.set(key, value)

    def get_image_size(self) -> Tuple[int, int]:
        return self.image_size

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self.set(name, value)

    def __getattr__(self, name: str) -> Any:
        if name == "fields" or name not in self.fields:
            raise AttributeError(f"Cannot find field \"{name}\" in the given Instances!")

        return self.fields[name]

    def set(self, name: str, value: Any) -> None:
        data_len = len(value)

        if len(self.fields):
            assert (len(self) == data_len
                    ), f"Adding a field of length {data_len} to a Instances of length {len(self)}"

        self.fields[name] = value

    def has(self, name: str) -> bool:
        return name in self.fields

    def remove(self, name: str) -> None:
        del self.fields[name]

    def get(self, name: str) -> Any:
        return self.fields[name]

    def get_field(self) -> Dict[str, Any]:
        return self.fields

    def to(self, *args: Any, **kwargs: Any) -> "Instances":
        result = Instances(self.image_size)

        for key, value in self.fields.items():
            if hasattr(value, "to"):
                value = value.to(*args, **kwargs)

            result.set(key, value)

        return result

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Instances":
        if isinstance(item, int):
            if item >= len(self) or item < -len(self):
                raise IndexError("Instances index out of range!")
            else:
                item = slice(item, None, len(self))

        result = Instances(self.image_size)

        for key, value in self.fields.items():
            result.set(key, value[item])

        return result

    def __len__(self) -> int:
        for value in self.fields.values():
            return len(value)

        raise NotImplementedError("Empty Instances does not support __len__!")

    def __iter__(self):
        raise NotImplementedError("\"Instances\" object is not iterable!")

    @staticmethod
    def cat(instance_lists: List["Instances"]) -> "Instances":
        assert all(isinstance(instance, Instances) for instance in instance_lists)
        assert len(instance_lists) > 0

        if len(instance_lists) == 1:
            return instance_lists[0]

        image_size = instance_lists[0].image_size

        for instance in instance_lists[1:]:
            assert instance.image_size == image_size

        result = Instances(image_size)

        for key in instance_lists[0].fields.keys():
            values = [instance.get(key) for instance in instance_lists]
            value_0 = values[0]

            if isinstance(value_0, torch.Tensor):
                values = torch.cat(values, dim=0)
            elif isinstance(value_0, list):
                values = list(itertools.chain(*values))
            elif hasattr(type(value_0), "cat"):
                values = type(value_0).cat(values)
            else:
                raise ValueError(f"Unsupported type {type(value_0)} for concatenation")

            result.set(key, values)

        return result

    def __str__(self) -> str:
        string = (f"num_instances={len(self)}, "
                  f"image_height={self.image_size[0]}, "
                  f"image_width={self.image_size[1]}, "
                  f"fields=[{','.join((f'{key}: {value}' for key, value in self.fields.items()))}]")

        return f"{self.__class__.__name__}({string})"
