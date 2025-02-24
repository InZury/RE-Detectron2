import functools
import logging
import os
import sys
import time

from sys import _getframe
from collections import Counter
from fvcore.common.file_io import PathManager
from tabulate import tabulate
from termcolor import colored

LOG_COUNTER = Counter()
LOG_TIMER = {}


class ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self.root_name = kwargs.pop("root_name") + "."
        self.abbrev_name = kwargs.pop("abbrev_name", "")

        if len(self.abbrev_name):
            self.abbrev_name = self.abbrev_name + "."

        super(ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self.root_name, self.abbrev_name)
        log = super(ColorfulFormatter, self).formatMessage(record)

        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "light_red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log

        return f"{prefix} {log}"


@functools.lru_cache()
def setup_logger(output=None, distributed_rank=0, *, color=True, name="re_detectron2", abbrev_name=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if abbrev_name is None:
        abbrev_name = "re_d2" if name == "re_detectron2" else name

    plain_formatter = logging.Formatter("[{asctime}] {name} {levelname} {message}", datefmt="%m/%d %H:%M:%S", style='{')

    if distributed_rank == 0:
        color_handler = logging.StreamHandler(stream=sys.stdout)
        color_handler.setLevel(logging.DEBUG)

        if color:
            formatter = ColorfulFormatter(
                colored("[{asctime} {name}]: ", "green") + "{message}",
                datefmt="%m/%d %H:%M:%S",
                root_name=name,
                abbrev_name=str(abbrev_name)
            )
        else:
            formatter = plain_formatter

        color_handler.setFormatter(formatter)
        logger.addHandler(color_handler)

    if output is not None:
        if output.endswitch(".txt") or output.endswitch(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")

        if distributed_rank > 0:
            filename = filename + f".rank{distributed_rank}"

        PathManager.mkdirs(os.path.dirname(filename))

        file_handler = logging.StreamHandler(cache_log_stream(filename))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(plain_formatter)
        logger.addHandler(file_handler)

    return logger


@functools.lru_cache(maxsize=None)
def cache_log_stream(filename):
    return PathManager.open(filename, "a")


def find_caller():
    frame = _getframe(2)

    while frame:
        code = frame.f_code

        if os.path.join("utils", "logger.") not in code.co_filename:
            mod_name = frame.f_globals["__name__"]

            if mod_name == "__main__":
                mod_name = "re_detectron2"

            return mod_name, (code.co_filename, frame.f_lineno, code.co_name)

        frame = frame.f_back


def log_first_n(level, message, n=1, *, name=None, key="caller"):
    if isinstance(key, str):
        key = (key, )

    assert len(key) > 0

    caller_module, caller_key = find_caller()
    hash_key = ()

    if "caller" in key:
        hash_key += caller_key
    if "message" in key:
        hash_key += (message, )

    LOG_COUNTER[hash_key] += 1

    if LOG_COUNTER[hash_key] <= n:
        logging.getLogger(name or caller_module).log(level, message)


def log_every_n(level, message, n=1, *, name=None):
    caller_module, caller_key = find_caller()
    LOG_COUNTER[caller_key] += 1

    if n == 1 or LOG_COUNTER[caller_key] % n == 1:
        logging.getLogger(name or caller_module).log(level, message)


def log_every_n_seconds(level, message, n=1, *, name=None):
    caller_module, caller_key = find_caller()
    last_logged = LOG_TIMER.get(caller_key, None)
    current_time = time.time()

    if not isinstance(last_logged, (float, int)):
        last_logged = 0.0

    if last_logged is None or current_time - last_logged >= n:
        logging.getLogger(name or caller_module).log(level, message)
        LOG_TIMER[caller_key] = current_time


def create_small_table(small_dict):
    keys, values = tuple(zip(*small_dict.items()))
    table = tabulate(
        tabular_data=[values],
        headers=keys,
        tablefmt="pipe",
        floatfmt=".3f",
        stralign="center",
        numalign="center"
    )

    return table
