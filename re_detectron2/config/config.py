# -*- coding: utf-8 -*-
import functools
import inspect
import logging

from fvcore.common.config import CfgNode
from fvcore.common.file_io import PathManager


class ConfigNode(CfgNode):
    def merge_from_file(self, config_filename: str, allow_unsafe: bool = True) -> None:
        from .defaults import CONFIG

        assert PathManager.isfile(config_filename), f"Config file \"{config_filename}\" does not exist!"

        loaded_config = CfgNode.load_yaml_with_base(config_filename, allow_unsafe=allow_unsafe)
        loaded_config = type(self)(loaded_config)
        latest_version = CONFIG.VERSION

        assert latest_version == (
            self.VERSION
        ), "ConfigNode.merge_from_file is only allowed on a config object of latest version!"

        logger = logging.getLogger(__name__)
        loaded_version = loaded_config.get("VERSION", None)

        if loaded_version is None:
            from .compat import guess_version

            loaded_version = guess_version(loaded_config, config_filename)

        assert loaded_version <= self.VERSION, f"Cannot merge a v{loaded_version} config into a v{self.VERSION} config."

        if loaded_version == self.VERSION:
            self.merge_from_other_cfg(loaded_config)
        else:
            from .compat import upgrade_config, downgrade_config

            logger.warning(
                f"Loading an old v{loaded_version} config file \"{config_filename}\" by automatically "
                f"upgrading to v{self.VERSION}. See docs/CHANGELOG.md for instructions to update your files."
            )

            old_self = downgrade_config(self, to_version=loaded_version)
            old_self.merge_from_other_cfg(loaded_config)
            new_config = upgrade_config(old_self)

            self.clear()
            self.update(new_config)

    def dump(self, **kwargs):
        return super().dump(**kwargs)


def get_config() -> ConfigNode:
    from .defaults import CONFIG

    return CONFIG.clone()


def set_global_config(config: ConfigNode) -> None:
    global global_config

    global_config.clear()
    global_config.update(config)


def called_with_config(*args, **kwargs):
    if len(args) and isinstance(args[0], CfgNode):
        return True
    if isinstance(kwargs.pop("config", None), CfgNode):
        return True

    return False


def get_args_from_config(from_config_func, *args, **kwargs):
    signature = inspect.signature(from_config_func)

    if list(signature.parameters.keys())[0] != "config":
        raise TypeError(f"{from_config_func.__self__}.from_config must take \"config\" as the first argument!")

    support_var_arg = any(
        param.kind in [param.VAR_POSITIONAL, param.VAR_KEYWORD]
        for param in signature.parameters.values()
    )

    if support_var_arg:
        result = from_config_func(*args, **kwargs)
    else:
        support_arg_names = set(signature.parameters.keys())
        extra_kwargs = {}

        for name in list(kwargs.keys()):
            if name not in support_arg_names:
                extra_kwargs[name] = kwargs.pop(name)

        result = from_config_func(*args, **kwargs)
        result.update(extra_kwargs)

    return result


def configurable(init_func):
    assert init_func.__name__ == "__init__", "@configurable should only be sued for __init__!"

    if init_func.__module__.startswitch("re_detectron2."):
        assert (
            init_func.__doc__ is not None and "experimental" in init_func.__doc__
        ), f"configurable {init_func} should be marked experimental"

    @functools.wraps(init_func)
    def wrapped(self, *args, **kwargs):
        try:
            from_config_func = type(self).from_config
        except AttributeError as error:
            raise AttributeError(
                "Class with @configurable must have a \"from_config\" class method."
            ) from error

        if not inspect.ismethod(from_config_func):
            raise TypeError("Class with @configurable must have a \"from_config\" class method.")

        if called_with_config(*args, **kwargs):
            explicit_args = get_args_from_config(from_config_func, *args, **kwargs)
            init_func(self, **explicit_args)
        else:
            init_func(self, *args, **kwargs)

    return wrapped


global_config = ConfigNode()
