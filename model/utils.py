import logging
import math

import numpy as np
import torch
from pytorch_lightning.utilities import rank_zero_only

from model.layers import BesselBasis, GaussianRBF, get_weight_init_by_string, str2act
from model.refnet import log
from ase.data import atomic_masses


def get_center_of_mass_torch(atomic_numbers, positions):
    """
    Computes center of mass.
    Args:
        atoms (ase.Atoms): atoms object of molecule
    Returns:
        center of mass
    """
    torch_masses = torch.tensor(
        atomic_masses, device=atomic_numbers.device, dtype=torch.float32
    )
    masses = torch_masses[atomic_numbers]
    return masses[:, None].T @ positions / masses.sum()


def get_center_of_mass(atomic_numbers, positions):
    """
    Computes center of mass.
    Args:
        atoms (ase.Atoms): atoms object of molecule
    Returns:
        center of mass
    """
    masses = atomic_masses[atomic_numbers]
    return np.dot(masses, positions) / masses.sum()


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


log = get_logger(__name__)


def _split(merged, num_vec):
    """
    Splits a merged representation of (s, V) back into a tuple.

    This function should be used in conjunction with `_merge(s, V)` and
    only if the tuple representation cannot be used.

    Args:
        merged (torch.Tensor): The tensor returned from `_merge`.
        num_vec (int): The number of vector channels in the input to `_merge`.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of split scalar and vector tensors.
    """
    v = torch.reshape(merged[..., -3 * num_vec :], merged.shape[:-1] + (3, num_vec))
    s = merged[..., : -3 * num_vec]
    return s, v


def _merge(s, V):
    """
    Merges a tuple (s, V) into a single `torch.Tensor`.

    In this process, the vector channels are flattened and appended to
    the scalar channels. It should be used only if the tuple representation
    cannot be used. Use `_split(x, nv)` to reverse the operation.

    Args:
        s (torch.Tensor): Scalar tensor.
        V (torch.Tensor): Vector tensor.

    Returns:
        torch.Tensor: Merged tensor.
    """
    V = torch.reshape(V, V.shape[:-2] + (3 * V.shape[-1],))
    return torch.cat([s, V], -1)


def _init(activation, bias_init, weight_init):
    """
    Initializes weights, biases and the activation function.

    Args:
    activation (str): Name of the activation function to use.
    bias_init (str): Name of the bias initialization method to use.
    weight_init (str): Name of the weight initialization method to use.

    Returns:
    Tuple[Callable, Callable, Callable]: Initialized activation function, bias initializer, and weight initializer.
    """
    if type(weight_init) == str:
        log.info(f"Using {weight_init} weight initialization")
        weight_init = get_weight_init_by_string(weight_init)
    if type(bias_init) == str:
        bias_init = get_weight_init_by_string(bias_init)
    if type(activation) is str:
        activation = str2act(activation)
    return activation, bias_init, weight_init


def basis(basis):
    """
    Determines the type of radial basis function to use.

    Args:
    basis (str): Input basis type ("BesselBasis" or "GaussianRBF").

    Returns:
    Callable: Selected basis function.

    Raises:
    ValueError: If an unknown radial basis is provided.
    """
    if basis == "BesselBasis":
        basis = BesselBasis
    elif basis == "GaussianRBF":
        basis = GaussianRBF
    else:
        raise ValueError("Unknown radial basis: {}".format(basis))
    return basis


INV_SQRT_3 = 1 / math.sqrt(3)
