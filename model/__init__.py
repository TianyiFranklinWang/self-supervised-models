from .mae import *
from .byol import *
from .dino import *
from .registry import *


def create_model(model_name, image_size, **kwargs):
    if not is_model(model_name):
        raise RuntimeError(f'Unknown model ({model_name})')

    create_fn = model_entrypoint(model_name)
    model = create_fn(image_size=image_size, **kwargs)

    return model
