from .mae import *
from .byol import *
from .dino import *
from .registry import *


def create_model(model_name, **kwargs):
    if not is_model(model_name):
        raise RuntimeError(f'Unknown model ({model_name})')

    create_fn = model_entrypoint(model_name)
    model = create_fn(**kwargs)

    return model
