# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import functools
import inspect
import logging
from fvcore.common.config import CfgNode as _CfgNode
from fvcore.common.file_io import PathManager


class CfgNode(_CfgNode):

    # Note that the default value of allow_unsafe is changed to True
    def merge_from_file(self, cfg_filename: str, allow_unsafe: bool = True) -> None:
        assert PathManager.isfile(cfg_filename), f"Config file '{cfg_filename}' does not exist!"
        loaded_cfg = _CfgNode.load_yaml_with_base(cfg_filename, allow_unsafe=allow_unsafe)
        loaded_cfg = type(self)(loaded_cfg)
        # defaults.py needs to import CfgNode
        from .defaults import _C
        
        self.merge_from_other_cfg(loaded_cfg)

    def dump(self, *args, **kwargs):
        """
        Returns:
            str: a yaml string representation of the config
        """
        # to make it show up in docs
        return super().dump(*args, **kwargs)


global_cfg = CfgNode()


def get_cfg() -> CfgNode:
    """
    Get a copy of the default config.

    Returns:
        a detectron2 CfgNode instance.
    """
    from .defaults import _C

    return _C.clone()


def set_global_cfg(cfg: CfgNode) -> None:
    """
    Let the global config point to the given cfg.

    Assume that the given "cfg" has the key "KEY", after calling
    `set_global_cfg(cfg)`, the key can be accessed by:
    ::
        from detectron2.config import global_cfg
        print(global_cfg.KEY)

    By using a hacky global config, you can access these configs anywhere,
    without having to pass the config object or the values deep into the code.
    This is a hacky feature introduced for quick prototyping / research exploration.
    """
    global global_cfg
    global_cfg.clear()
    global_cfg.update(cfg)


def configurable(init_func):
    """
    Decorate a class's __init__ method so that it can be called with a CfgNode
    object using the class's from_config classmethod.

    Examples:
    ::
        class A:
            @configurable
            def __init__(self, a, b=2, c=3):
                pass

            @classmethod
            def from_config(cls, cfg):
                # Returns kwargs to be passed to __init__
                return {"a": cfg.A, "b": cfg.B}

        a1 = A(a=1, b=2)  # regular construction
        a2 = A(cfg)       # construct with a cfg
        a3 = A(cfg, b=3, c=4)  # construct with extra overwrite
    """
    assert init_func.__name__ == "__init__", "@configurable should only be used for __init__!"
    if init_func.__module__.startswith("detectron2."):
        assert (
            init_func.__doc__ is not None and "experimental" in init_func.__doc__
        ), f"configurable {init_func} should be marked experimental"

    @functools.wraps(init_func)
    def wrapped(self, *args, **kwargs):
        try:
            from_config_func = type(self).from_config
        except AttributeError:
            raise AttributeError("Class with @configurable must have a 'from_config' classmethod.")
        if not inspect.ismethod(from_config_func):
            raise TypeError("Class with @configurable must have a 'from_config' classmethod.")

        if _called_with_cfg(*args, **kwargs):
            explicit_args = _get_args_from_config(from_config_func, *args, **kwargs)
            init_func(self, **explicit_args)
        else:
            init_func(self, *args, **kwargs)

    return wrapped


def _get_args_from_config(from_config_func, *args, **kwargs):
    """
    Use `from_config` to obtain explicit arguments.

    Returns:
        dict: arguments to be used for cls.__init__
    """
    signature = inspect.signature(from_config_func)
    if list(signature.parameters.keys())[0] != "cfg":
        raise TypeError(
            f"{from_config_func.__self__}.from_config must take 'cfg' as the first argument!"
        )
    support_var_arg = any(
        param.kind in [param.VAR_POSITIONAL, param.VAR_KEYWORD]
        for param in signature.parameters.values()
    )
    if support_var_arg:  # forward all arguments to from_config, if from_config accepts them
        ret = from_config_func(*args, **kwargs)
    else:
        # forward supported arguments to from_config
        supported_arg_names = set(signature.parameters.keys())
        extra_kwargs = {}
        for name in list(kwargs.keys()):
            if name not in supported_arg_names:
                extra_kwargs[name] = kwargs.pop(name)
        ret = from_config_func(*args, **kwargs)
        # forward the other arguments to __init__
        ret.update(extra_kwargs)
    return ret


def _called_with_cfg(*args, **kwargs):
    """
    Returns:
        bool: whether the arguments contain CfgNode and should be considered
            forwarded to from_config.
    """
    if len(args) and isinstance(args[0], _CfgNode):
        return True
    if isinstance(kwargs.pop("cfg", None), _CfgNode):
        return True
    # `from_config`'s first argument is forced to be "cfg".
    # So the above check covers all cases.
    return False
