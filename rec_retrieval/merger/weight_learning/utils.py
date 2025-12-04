from ..types import *


def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])


def set_attr(obj, names, val, shape):
    if len(names) == 1:
        setattr(obj, names[0], val.reshape(shape))
    else:
        set_attr(getattr(obj, names[0]), names[1:], val, shape)


def make_functional(mod):
    orig_params = []
    names = []
    for name, p in list(mod.named_parameters()):
        orig_params.append(p.data.clone())
        names.append(name)
        del_attr(mod, name.split("."))

    return tuple(orig_params), names


def get_state_dict(params, shape_dict) -> StateDict:
    start_idx = 0
    state_dict = {}

    for name, shape in shape_dict.items():
        numel = shape.numel()
        p = params[start_idx : start_idx + numel]
        start_idx += numel
        state_dict[name] = p.reshape(shape)

    assert start_idx == len(params), "Not all parameters are loaded."
    return state_dict


def load_weights(mod, params, shape_dict):
    start_idx = 0

    for name, shape in shape_dict.items():
        p = params[start_idx : start_idx + shape.numel()]
        start_idx += shape.numel()
        set_attr(mod, name.split("."), p, shape)

    assert start_idx == len(params), "Not all parameters are loaded."
