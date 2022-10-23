_model_entrypoints = {}


def register_model(fn):
    """
    Decorator for registering a function in model entrypoint.
    """
    _model_entrypoints[fn.__name__] = fn
    return fn


def is_model(model_name):
    """
    Check if a model name exists.
    """
    return model_name in _model_entrypoints.keys()


def model_entrypoint(model_name):
    """
    Fetch a model entrypoint for specified model name.
    """
    return _model_entrypoints[model_name]


def list_models():
    return sorted(list(_model_entrypoints.keys()))
