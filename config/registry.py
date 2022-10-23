_config_entrypoints = {}


def register_config(cls):
    """
    Decorator for registering a function in model entrypoint.
    """
    _config_entrypoints[cls.__name__] = cls
    return cls


def is_config(config_name):
    """
    Check if a model name exists.
    """
    return config_name in _config_entrypoints.keys()


def config_entrypoint(config_name):
    """
    Fetch a model entrypoint for specified model name.
    """
    return _config_entrypoints[config_name]


def list_configs():
    return sorted(list(_config_entrypoints.keys()))
