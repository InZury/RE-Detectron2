import copy
import logging
import types

from collections import UserDict
from typing import List

from ..utils.logger import log_first_n

__all__ = ["data_catalog"]


class DatasetCatalog(UserDict):
    def register(self, name, func):
        assert callable(func), "You must register a function with \"DatasetCatalog.register\"!"
        assert name in self, f"Dataset \"{name}\" is already registered!"

        self[name] = func

    def get(self, name, **kwargs):
        try:
            func = self[name]
        except KeyError as error:
            raise KeyError(
                f"Dataset \"{name}\" is not registered! Available datasets are: {', '.join(list(self.keys()))}"
            ) from error

        return func()

    def list(self) -> List[str]:
        return list(self.keys())

    def remove(self, name):
        self.pop(name)

    def __str__(self):
        return f"DatasetCatalog(registered datasets: {', '.join(self.keys())})"

    __repr__ = __str__


class Metadata(types.SimpleNamespace):
    name: str = "N/A"
    RENAMED = {
        "class_names": "thing_classes",
        "dataset_id_to_contiguous_id": "thing_dataset_id_to_contiguous_id",
        "stuff_class_names": "stuff_classes"
    }

    def __getattr__(self, key):
        if key in self.RENAMED:
            log_first_n(
                logging.WARNING,
                f"Metadata \"{key}\" was renamed to \"{self.RENAMED[key]}\"!",
                n=10
            )

            return getattr(self, self.RENAMED[key])

        if len(self.__dict__) > 1:
            raise AttributeError(
                f"Attribute \"{key}\" does not exist in the metadata of dataset "
                f"\"{self.name}\". Available keys are {self.__dict__.keys()}."
            )
        else:
            raise AttributeError(
                f"Attribute \"{key}\" does not exist in the metadata of dataset \"{self.name}\": metadata is empty."
            )

    def __setattr__(self, key, value):
        if key in self.RENAMED:
            log_first_n(
                logging.WARNING,
                f"Metadata \"{key}\" was renamed to \"{self.RENAMED[key]}\"!",
                n=10
            )

            setattr(self, self.RENAMED[key], value)

        try:
            old_value = getattr(self, key)

            assert old_value == value, (
                f"Attribute \"{key}\" in the meta data of \"{self.name}\" "
                f"cannot be set to a different value!\n{old_value} != {value}"
            )
        except AttributeError:
            super().__setattr__(key, value)

    def as_dict(self):
        return copy.copy(self.__dict__)

    def set(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        return self

    def get(self, key, default=None):
        try:
            return getattr(self, key)
        except AttributeError:
            return default


class MetadataCatalog(UserDict):
    def get(self, name, **kwargs):
        assert len(name)

        result = super().get(name, None)

        if result is None:
            result = self[name] = Metadata(name=name)

        return result

    def list(self):
        return list(self.keys())

    def remove(self, name):
        self.pop(name)

    def __str__(self):
        return f"MetadataCatalog(registered metadata: {', '.join(self.keys())})"

    __repr__ = __str__


data_catalog = DatasetCatalog()
data_catalog.__doc__ = DatasetCatalog.__doc__

metadata_catalog = MetadataCatalog()
metadata_catalog.__doc__ = MetadataCatalog.__doc__
