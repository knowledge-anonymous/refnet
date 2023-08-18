""" RefNet Layers """

import inspect
import math
from functools import partial
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, functional
from torch import nn as nn
from torch.nn.init import constant_, xavier_uniform_
from torch_cluster import radius_graph
from torch_geometric.nn.inits import glorot_orthogonal

zeros_initializer = partial(constant_, val=0.0)


class LeakyCosineCutoff(nn.Module):
    r"""Class of Behler cosine cutoff.

    .. math::
       f(r) = \begin{cases}
        0.5 \times \left[1 + \cos\left(\frac{\pi r}{r_\text{cutoff}}\right)\right]
          & r < r_\text{cutoff} \\
        0 & r \geqslant r_\text{cutoff} \\
        \end{cases}

    Args:
        cutoff (float, optional): cutoff radius.

    """

    def __init__(self, cutoff=5.0, eps=0.1):
        super(LeakyCosineCutoff, self).__init__()
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))
        self.register_buffer("eps", torch.FloatTensor([eps]))

    def forward(self, distances):
        """Compute cutoff.

        Args:
            distances (torch.Tensor): values of interatomic distances.

        Returns:
            torch.Tensor: values of cutoff function.

        """
        # Compute values of cutoff function
        cutoffs = 0.5 * (torch.cos(distances * np.pi / self.cutoff) + 1.0)
        # Remove contributions beyond the cutoff radius
        cutoffs *= (distances < self.cutoff).float()
        return cutoffs + self.eps


@torch.jit.script
def safe_norm(x: Tensor, dim: int = -2, eps: float = 1e-8, keepdim: bool = False):
    return torch.sqrt(torch.sum(x**2, dim=dim, keepdim=keepdim)) + eps


def safe_norm_v2(x, dim=-2, eps=1e-8, keepdim=False, sqrt=True):
    out = torch.sum(x**2, dim=dim, keepdim=keepdim)
    if sqrt:
        out = torch.sqrt(out)

    return out + eps


def safest_norm(x, dim=-1, keepdim=False, eps=1e-8, sqrt=True):
    """
    L2 norm of tensor clamped above a minimum value `eps`.

    :param sqrt: if `False`, returns the square of the L2 norm
    """
    out = torch.clamp(
        torch.sum(torch.square(x), dim=dim, keepdim=keepdim) + eps, min=eps
    )
    return torch.sqrt(out) if sqrt else out


class ScaleShift(nn.Module):
    r"""Scale and shift layer for standardization.

    .. math::
       y = x \times \sigma + \mu

    Args:
        mean (torch.Tensor): mean value :math:`\mu`.
        stddev (torch.Tensor): standard deviation value :math:`\sigma`.

    """

    def __init__(self, mean, stddev):
        super(ScaleShift, self).__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("stddev", stddev)

    def forward(self, input):
        """Compute layer output.

        Args:
            input (torch.Tensor): input data.

        Returns:
            torch.Tensor: layer output.

        """
        y = input * self.stddev + self.mean
        return y


class GetItem(nn.Module):
    """Extraction layer to get an item from SchNetPack dictionary of input tensors.
    Args:
        key (str): Property to be extracted from SchNetPack input tensors.
    """

    def __init__(self, key):
        super(GetItem, self).__init__()
        self.key = key

    def forward(self, inputs):
        """Compute layer output.
        Args:
            inputs (dict of torch.Tensor): SchNetPack dictionary of input tensors.
        Returns:
            torch.Tensor: layer output.
        """
        return inputs[self.key]


# @torch.jit.script
def xyz_to_feat(pos: Tensor, edge_index: List[Tensor]):
    j, i = edge_index  # j->i

    # Calculate distances. # number of edges
    dist_vec = pos[i] - pos[j]
    # distance = dist_vec.pow(2).sum(dim=-1).sqrt()
    # distance = torch.norm(dist_vec, 2, -1)
    distance = safe_norm(dist_vec, dim=-1)

    dist_vec_norm = dist_vec / distance.unsqueeze(-1)

    return distance, dist_vec_norm


def get_geometry(batch, cutoff=5.0):
    atomic_numbers, pos, batch_idx = batch.z, batch.pos, batch.batch

    if "edge_index" not in batch:
        edge_index = radius_graph(pos, r=cutoff, loop=False, batch=batch_idx)
        batch.edge_index = edge_index
    else:
        edge_index = batch.edge_index

    if "dir_ij" not in batch or "rij" not in batch:
        rij, dir_ij = xyz_to_feat(pos, edge_index=edge_index, num_nodes=pos.size(0))
        batch.rij = rij
        batch.dir_ij = dir_ij

    return batch


def gaussian_rbf(inputs: torch.Tensor, offsets: torch.Tensor, widths: torch.Tensor):
    coeff = -0.5 / torch.pow(widths, 2)
    diff = inputs[..., None] - offsets
    y = torch.exp(coeff * torch.pow(diff, 2))
    return y


class GaussianRBF(nn.Module):
    r"""Gaussian radial basis functions."""

    def __init__(
        self, n_rbf: int, cutoff: float, start: float = 0.0, trainable: bool = False
    ):
        """
        Args:
            n_rbf: total number of Gaussian functions, :math:`N_g`.
            cutoff: center of last Gaussian function, :math:`\mu_{N_g}`
            start: center of first Gaussian function, :math:`\mu_0`.
            trainable: If True, widths and offset of Gaussian functions
                are adjusted during training process.
        """
        super(GaussianRBF, self).__init__()
        self.n_rbf = n_rbf

        # compute offset and width of Gaussian functions
        offset = torch.linspace(start, cutoff, n_rbf)
        widths = torch.FloatTensor(
            torch.abs(offset[1] - offset[0]) * torch.ones_like(offset)
        )
        if trainable:
            self.widths = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("widths", widths)
            self.register_buffer("offsets", offset)

    def forward(self, inputs: torch.Tensor):
        return gaussian_rbf(inputs, self.offsets, self.widths)


class BesselBasis(nn.Module):
    """
    Sine for radial basis expansion with coulomb decay. (0th order Bessel from DimeNet)
    """

    def __init__(self, cutoff=5.0, n_rbf=None):
        """
        Args:
            cutoff: radial cutoff
            n_rbf: number of basis functions.
        """
        super(BesselBasis, self).__init__()
        # compute offset and width of Gaussian functions
        freqs = torch.arange(1, n_rbf + 1) * math.pi / cutoff
        self.register_buffer("freqs", freqs)
        self.register_buffer("norm1", torch.tensor(1.0))

    def forward(self, inputs):
        input_size = len(inputs.shape)
        a = self.freqs[None, :]
        inputs = inputs[..., None]
        ax = inputs * a
        sinax = torch.sin(ax)

        norm = torch.where(inputs == 0, self.norm1, inputs)
        y = sinax / norm

        return y


def glorot_orthogonal_wrapper_(tensor, scale=2.0):
    return glorot_orthogonal(tensor, scale=scale)


def _standardize(kernel):
    """
    Makes sure that Var(W) = 1 and E[W] = 0
    """
    eps = 1e-6

    if len(kernel.shape) == 3:
        axis = [0, 1]  # last dimension is output dimension
    else:
        axis = 1

    var, mean = torch.var_mean(kernel, dim=axis, unbiased=True, keepdim=True)
    kernel = (kernel - mean) / (var + eps) ** 0.5
    return kernel


def he_orthogonal_init(tensor):
    """
    Generate a weight matrix with variance according to He initialization.
    Based on a random (semi-)orthogonal matrix neural networks
    are expected to learn better when features are decorrelated
    (stated by eg. "Reducing overfitting in deep networks by decorrelating representations",
    "Dropout: a simple way to prevent neural networks from overfitting",
    "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks")
    """
    tensor = torch.nn.init.orthogonal_(tensor)

    if len(tensor.shape) == 3:
        fan_in = tensor.shape[:-1].numel()
    else:
        fan_in = tensor.shape[1]

    with torch.no_grad():
        tensor.data = _standardize(tensor.data)
        tensor.data *= (1 / fan_in) ** 0.5

    return tensor


def get_weight_init_by_string(init_str):
    if init_str == "":
        # Noop
        return lambda x: x
    elif init_str == "zeros":
        return torch.nn.init.zeros_
    elif init_str == "xavier_uniform":
        return torch.nn.init.xavier_uniform_
    elif init_str == "glo_orthogonal":
        return glorot_orthogonal_wrapper_
    elif init_str == "he_orthogonal":
        return he_orthogonal_init
    else:
        raise ValueError(f"Unknown initialization {init_str}")


class Dense(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        activation=None,
        weight_init=xavier_uniform_,
        bias_init=zeros_initializer,
        norm=None,
        gain=None,
    ):
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.gain = gain
        super(Dense, self).__init__(in_features, out_features, bias)
        # Initialize activation function
        if inspect.isclass(activation):
            self.activation = activation()
        self.activation = activation

        if norm == "layer":
            self.norm = nn.LayerNorm(out_features)
        elif norm == "batch":
            self.norm = nn.BatchNorm1d(out_features)
        elif norm == "instance":
            self.norm = nn.InstanceNorm1d(out_features)
        else:
            self.norm = None

    def reset_parameters(self):
        """Reinitialize model weight and bias values."""
        if self.gain:
            self.weight_init(self.weight, gain=self.gain)
        else:
            self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, inputs):
        y = super(Dense, self).forward(inputs)
        if self.norm is not None:
            y = self.norm(y)
        if self.activation:
            y = self.activation(y)
        return y


class MLP(nn.Module):
    def __init__(
        self,
        hidden_dims: List[int],
        n_layers: int = -1,
        bias=True,
        activation=None,
        last_activation=None,
        weight_init=xavier_uniform_,
        bias_init=zeros_initializer,
    ):
        super().__init__()

        DenseMLP = partial(
            Dense, bias=bias, weight_init=weight_init, bias_init=bias_init
        )

        if n_layers > 1:
            assert (
                len(hidden_dims) == 3
            ), "n_layers and hidden_dims are mutually exclusive"
            dim_in, dim_hid, dim_out = hidden_dims
            hidden_dims = [dim_in] + [dim_hid] * (n_layers - 1) + [dim_out]

        dims = hidden_dims
        n_layers = len(dims)

        self.dense_layers = nn.ModuleList(
            [
                DenseMLP(dims[i], dims[i + 1], activation=activation)
                for i in range(n_layers - 2)
            ]
            + [DenseMLP(dims[-2], dims[-1], activation=last_activation)]
        )

        self.layers = nn.Sequential(*self.dense_layers)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.dense_layers:
            m.reset_parameters()

    def forward(self, x):
        return self.layers(x)


class _VDropout(nn.Module):
    """
    Vector channel dropout where the elements of each
    vector channel are dropped together.
    """

    def __init__(self, drop_rate, scale=True):
        super(_VDropout, self).__init__()
        self.drop_rate = drop_rate
        self.scale = scale

    def forward(self, x, dim=-1):
        """
        :param x: `torch.Tensor` corresponding to vector channels
        """
        if self.drop_rate == 0:
            return x
        device = x.device
        if not self.training:
            return x

        shape = list(x.shape)
        assert shape[dim] == 3, "The dimension must be vector"
        shape[dim] = 1

        mask = torch.bernoulli((1 - self.drop_rate) * torch.ones(shape, device=device))
        x = mask * x
        if self.scale:
            # scale the output to keep the expected output distribution
            # same as input distribution. However, this might be harmfuk
            # for vector space.
            x = x / (1 - self.drop_rate)

        return x


def normalize_string(s: str) -> str:
    return s.lower().replace("-", "").replace("_", "").replace(" ", "")


def shifted_softplus(x: torch.Tensor):
    return functional.softplus(x) - math.log(2.0)


def scaled_silu(x, scale=0.6):
    return F.silu(x) * scale


class ScaledSiLU(torch.nn.Module):
    """
    Scaled SiLU activation function.
    """

    def __init__(self):
        super().__init__()
        self.scale_factor = 1 / 0.6
        self._activation = torch.nn.SiLU()

    def forward(self, x):
        return scaled_silu(x, self.scale_factor)


def get_activations(optional=False, *args, **kwargs):
    """
    Get all available activation functions.
    Based on https://github.com/sunglasses-ai/classy/blob/3e74cba1fdf1b9f9f2ba1cfcfa6c2017aa59fc04/classy/optim/factories.py#L14

    Args:
        optional:
        *args: positional arguments to be passed to the activation function
        **kwargs: argument dictionary to be passed to the activation function

    Returns:

    """
    activations = {
        normalize_string(act.__name__): act
        for act in vars(torch.nn.modules.activation).values()
        if isinstance(act, type) and issubclass(act, torch.nn.Module)
    }

    activations.update(
        {
            "relu": torch.nn.ReLU,
            "elu": torch.nn.ELU,
            "sigmoid": torch.nn.Sigmoid,
            "silu": torch.nn.SiLU,
            "swish": torch.nn.SiLU,
            "selu": torch.nn.SELU,
            "scaled_swish": scaled_silu,
            "softplus": shifted_softplus,
        }
    )

    if optional:
        activations[""] = None

    return activations


def dictionary_to_option(options, selected):
    if selected not in options:
        raise ValueError(
            f'Invalid choice "{selected}", choose one from {", ".join(list(options.keys()))} '
        )

    activation = options[selected]
    if inspect.isclass(activation):
        activation = activation()
    return activation


def str2act(input_str, *args, **kwargs):
    if input_str == "":
        return None

    act = get_activations(optional=True, *args, **kwargs)
    out = dictionary_to_option(act, input_str)
    return out
