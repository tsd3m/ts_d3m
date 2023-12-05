import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Dict, List, Optional
import warnings
from functools import partial


def gelu_accurate(x):
    if not hasattr(gelu_accurate, "_a"):
        gelu_accurate._a = math.sqrt(2 / math.pi)
    return (
        0.5 * x * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 * torch.pow(x, 3))))
    )


def gelu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x.float()).type_as(x)


def deprecation_warning(message, stacklevel=3):
    # don't use DeprecationWarning, since it's ignored by default
    warnings.warn(message, stacklevel=stacklevel)

def get_activation_fn(activation: str) -> Callable:
    """ Returns the activation function corresponding to `activation` """
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return gelu
    elif activation == "gelu_fast":
        deprecation_warning(
            "--activation-fn=gelu_fast has been renamed to gelu_accurate"
        )
        return gelu_accurate
    elif activation == "gelu_accurate":
        return gelu_accurate
    elif activation == 'silu':
        return F.silu
    elif activation == "linear":
        return lambda x: x
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))
    

def relu2(x, onnx_trace: bool = False):
    if onnx_trace:
        return torch.square(F.relu(x.float()))
    else:
        return torch.square(F.relu(x))
    

def laplace(x, mu=0.707107, sigma=0.282095, onnx_trace: bool = False):
    if onnx_trace:
        x = x.float()
    x = (x - mu).div(sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + torch.erf(x))


def softmax(x, dim: int, onnx_trace: bool = False):
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    else:
        return F.softmax(x, dim=dim, dtype=torch.float32)


def Activation(activation=None, dim=-1):
    if activation in [ None, 'id', 'identity', 'linear' ]:
        return nn.Identity()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'elu':
        return nn.ELU()
    elif activation in ['swish', 'silu']:
        return nn.SiLU()
    elif activation == 'glu':
        return nn.GLU(dim=dim)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'softplus':
        return nn.Softplus()
    else:
        raise NotImplementedError("hidden activation '{}' is not implemented".format(activation))

def LinearActivation(
        d_input, d_output, bias=True,
        transposed=False,
        activation=None,
        activate=False, # Apply activation as part of this module
        **kwargs,
    ):
    """Returns a linear nn.Module with control over axes order, initialization, and activation."""

    # Construct core module
    linear_cls = partial(nn.Conv1d, kernel_size=1) if transposed else nn.Linear
    if activation is not None and activation == 'glu': d_output *= 2
    linear = linear_cls(d_input, d_output, bias=bias, **kwargs)

    if activate and activation is not None:
        activation = Activation(activation, dim=-2 if transposed else -1)
        linear = nn.Sequential(linear, activation)
    return linear
