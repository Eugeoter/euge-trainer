from typing import get_type_hints
from ml_collections import ConfigDict


class FromConfigMixin(object):
    @classmethod
    def from_config(cls, config, **kwargs):
        # for k, v in get_full_type_hints(cls).items():
        for k, v in config.items():
            kwargs.setdefault(k, config[k])
        return cls(config, **kwargs)

    def __init__(self, config, **kwargs):
        self.config = config
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get_config(self):
        return self.config


class SubModuleMixin(FromConfigMixin):
    @classmethod
    def from_module(cls, module, **kwargs):
        return cls(module, **kwargs)

    def __init__(self, module, **kwargs):
        self.module = module
        for attr_name in dir(self.module):
            attr = getattr(self.module, attr_name)
            if not attr_name.startswith('__') and not callable(attr) and hasattr(self, attr_name):
                setattr(self, attr_name, attr)

    def __getattr__(self, name):
        return getattr(self.module, name)


def get_full_type_hints(cls):
    vars = get_type_hints(cls)
    for base in reversed(cls.__mro__):
        vars.update({k: type(v) for k, v in base.__dict__.items() if not k.startswith('__') and not callable(v)})
    return vars


def cfg(**kwargs):
    return ConfigDict(initial_dictionary=kwargs)
