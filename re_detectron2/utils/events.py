import datetime
import logging
import json
import os
import time
import torch

from collections import defaultdict
from contextlib import contextmanager
from fvcore.common.file_io import PathManager
from fvcore.common.history_buffer import HistoryBuffer

__all__ = ["get_event_storage", "JSONWriter", "TensorboardXWriter", "CommonMetricPrinter", "EventStorage"]
CURRENT_STORAGE_STACK = []


class EventWriter:
    def write(self):
        raise NotImplementedError

    def close(self):
        pass


class JSONWriter(EventWriter):
    def __init__(self, json_file, window_size=20):
        self.file_handle = PathManager.open(json_file, "a")
        self.window_size = window_size
        self.last_write = -1

    def write(self):
        storage = get_event_storage()
        items = defaultdict(dict)

        for key, (value, iteration) in storage.latest_with_smoothing_hint(self.window_size).items():
            if iteration <= self.last_write:
                continue

            items[iteration][key] = value

        if len(items):
            all_iters = sorted(items.keys())
            self.last_write = max(all_iters)

        for iteration, scalars in items.items():
            scalars["iteration"] = iteration
            self.file_handle.write(json.dumps(scalars, sort_keys=True) + "\n")

        self.file_handle.flush()

        try:
            os.fsync(self.file_handle.fileno())
        except AttributeError:
            pass

    def close(self):
        self.file_handle.close()


class TensorboardXWriter(EventWriter):
    def __init__(self, log_dir: str, window_size: int = 20, **kwargs):
        from torch.utils.tensorboard import SummaryWriter

        self.window_size = window_size
        self.writer = SummaryWriter(log_dir, **kwargs)
        self.last_write = -1

    def write(self):
        storage = get_event_storage()
        new_last_write = self.last_write

        for key, (value, iteration) in storage.latest_with_smoothing_hint(self.window_size).items():
            if iteration > self.last_write:
                self.writer.add_scalar(key, value, iteration)
                new_last_write = max(new_last_write, iteration)

        self.last_write = new_last_write

        if len(storage.visual_data) >= 1:
            for image_name, image, step_num in storage.visual_data:
                self.writer.add_image(image_name, image, step_num)

            storage.clear_images()

        if len(storage.histograms) >= 1:
            for params in storage.histograms:
                self.writer.add_histogram_raw(**params)

            storage.clear_histograms()

    def close(self):
        if hasattr(self, "writer"):
            self.writer.close()


class CommonMetricPrinter(EventWriter):
    def __init__(self, max_iter):
        self.logger = logging.getLogger(__name__)
        self.max_iter = max_iter
        self.last_write = None

    def write(self):
        storage = get_event_storage()
        iteration = storage.iteration
        eta_string = None

        if iteration == self.max_iter:
            return

        try:
            data_time = storage.history("data_time").avg(20)
        except KeyError:
            data_time = None

        try:
            iter_time = storage.history("time").global_avg()
            eta_seconds = storage.history("time").median(1000) * (self.max_iter - iteration - 1)
            storage.put_scalar("eta_seconds", eta_seconds, smoothing_hint=False)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        except KeyError:
            iter_time = None

            if self.last_write is not None:
                estimate_iter_time = (time.perf_counter() - self.last_write[1]) / (iteration - self.last_write[0])
                eta_seconds = estimate_iter_time * (self.max_iter - iteration - 1)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            self.last_write = (iteration, time.perf_counter())

        try:
            learning_rate = f"{storage.history('learning_rate').latest():.5g}"
        except KeyError:
            learning_rate = "N/A"

        if torch.cuda.is_available():
            max_memory_size = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        else:
            max_memory_size = None

        self.logger.info(
            " {eta}iter: {iteration}  {losses}  {time}{data_time}lr: {learning_rate}  {memory}".format(
                eta=f"eta: {eta_string}  " if eta_string else "",
                iteration=iteration,
                losses="  ".join(
                    [
                        f"{key}: {value.median(20):.4g}"
                        for key, value in storage.histories().items()
                        if "loss" in key
                    ]
                ),
                time=f"time:{iter_time:.4f}" if iter_time is not None else "",
                data_time=f"data_time: {data_time:.4f}  " if data_time is not None else "",
                learning_rate=learning_rate,
                memory=f"max_memory: {max_memory_size:.0f}M" if max_memory_size is not None else "",
            )
        )


class EventStorage:
    def __init__(self, start_iter=0):
        self.history = defaultdict(HistoryBuffer)
        self.smoothing_hints = {}
        self.latest_scalars = {}
        self.iteration = start_iter
        self.current_prefix = ""
        self.visual_data = []
        self.histograms = []

    def put_image(self, image_name, image_tensor):
        self.visual_data.append((image_name, image_tensor, self.iteration))

    def put_scalar(self, name, value, smoothing_hint=True):
        name = self.current_prefix + name
        history = self.history[name]
        value = float(value)
        history.update(value, self.iteration)
        self.latest_scalars[name] = (value, self.iteration)
        existed_hint = self.smoothing_hints.get(name)

        if existed_hint is not None:
            assert existed_hint == smoothing_hint, (
                f"Scalar {name} was put with a different smoothing_hint!"
            )
        else:
            self.smoothing_hints[name] = smoothing_hint

    def put_scalars(self, *, smoothing_hint=True, **kwargs):
        for key, value in kwargs.items():
            self.put_scalar(key, value, smoothing_hint=smoothing_hint)

    def put_histogram(self, hist_name, hist_tensor, bins=1000):
        hist_min, hist_max = hist_tensor.min().item(), hist_tensor.max().item()
        hist_counts = torch.histc(hist_tensor, bins=bins)
        hist_edges = torch.linspace(start=hist_min, end=hist_max, steps=bins + 1, dtype=torch.float32)
        hist_params = dict(
            tag=hist_name,
            min=hist_min,
            max=hist_max,
            num=len(hist_tensor),
            sum=float(hist_tensor.sum()),
            sum_squares=float(torch.sum(hist_tensor ** 2)),
            bucket_limits=hist_edges[1:].tolist(),
            bucket_counts=hist_counts.tolist(),
            global_step=self.iteration
        )
        self.histograms.append(hist_params)

    def history(self, name):
        result = self.history.get(name, None)

        if result is None:
            raise KeyError(f"No history metric available for {name}!")

        return result

    def histories(self):
        return self.history

    def latest(self):
        return self.latest_scalars

    def latest_with_smoothing_hint(self, window_size=20):
        result = {}

        for key, (value, iteration) in self.latest_scalars.items():
            result[key] = (self.history[key].median(window_size) if self.smoothing_hints[key] else value, iteration)

        return result

    def smoothing_hints(self):
        return self.smoothing_hints

    def step(self):
        self.iteration += 1

    @property
    def iter(self):
        return self.iteration

    @iter.setter
    def iter(self, value):
        self.iteration = int(value)

    def __enter__(self):
        CURRENT_STORAGE_STACK.append(self)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert CURRENT_STORAGE_STACK[-1] == self

        CURRENT_STORAGE_STACK.pop()

    @contextmanager
    def name_scope(self, name):
        old_prefix = self.current_prefix
        self.current_prefix = name.rstrip("/") + "/"

        yield

        self.current_prefix = old_prefix

    def clear_images(self):
        self.visual_data = []

    def clear_histograms(self):
        self.histograms = []


def get_event_storage():
    assert len(CURRENT_STORAGE_STACK), (
        "get_event_storage() has to be called inside a \"with EventStorage(...)\" context!"
    )

    return CURRENT_STORAGE_STACK[-1]
