# coding=utf-8

"""  """
import os
import pickle
from collections import defaultdict

import pandas as pd
import torch
import torch_geometric as pyg
from torch_geometric.data import Data, Batch
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_dense_batch, add_self_loops
from torch_geometric.data import Data, Batch
from torch_scatter import scatter_add
from torch_scatter import scatter

import numpy as np
from tqdm import tqdm

from sainn.datamodules.qm9_datamodule import AtomsDataModule


def compute_angle_between_vectors(v1, v2, dim=-1, eps=1e-7):
    """
    Compute the angle between two vectors.

    Args:
        v1 (Tensor): First vector.
        v2 (Tensor): Second vector.
        dim (int, optional): The dimension to reduce. Defaults to -1.
        eps (float, optional): A small number to prevent division by zero. Defaults to 1e-7.

    Returns:
        Tensor: The angle between the two vectors.
    """
    v1_norm = v1 / (v1.norm(dim=dim, keepdim=True) + eps)
    v2_norm = v2 / (v2.norm(dim=dim, keepdim=True) + eps)
    dot_products = (v1_norm * v2_norm).sum(dim=dim)
    return torch.acos(dot_products.clamp(-1, 1))



def create_non_linear_mask(points, eps=1e-7, linear_threshold=1e-2):
    """
    Create a mask that is True for all triplets of points that form a non-linear configuration.

    The configuration is considered non-linear if the angle between the vectors formed by the points is above a certain threshold.

    Args:
        points: Input tensor. Shape: (B, K, 3), where B is the batch size, K is the number of points in each batch, and 3 is the dimension of each point.
        eps: Small positive number to avoid division by zero in the computation of the angle.
        linear_threshold: Threshold for the minimum angle (in radians) to consider the configuration as non-linear.

    Returns:
        non_linear_mask: A mask tensor. Shape: (B, K-2). It is True at positions corresponding to non-linear configurations of points.
    """
    # Compute the vectors between the first two points and all other points
    v1 = points[:, 0, :] - points[:, 1, :]  # Vector from first to second point, shape: B x 3
    v = points[:, 2:, :] - points[:, 1:2, :]  # Vectors from second to all other points, shape: B x (K-2) x 3

    # Compute the angle between the vectors v1 and v. Assuming compute_angle_between_vectors function is previously defined.
    angle = compute_angle_between_vectors(v1.unsqueeze(1), v, eps=eps)

    # Create a mask that is True for non-linear points
    # A configuration is considered non-linear if the angle is above the linear_threshold and below pi - linear_threshold
    non_linear_mask = ((angle > linear_threshold) & (angle < np.pi - linear_threshold))

    return non_linear_mask.squeeze(-1)


def compute_torsion_angles_ex(points, eps=1e-7):
    """
    Compute the torsion angles for a set of points.

    The first three points are fixed, and the torsion angle is computed relative to these.

    Args:
        points: Input tensor. Shape: (B, K, 3), where B is the batch size, K is the number of points in each batch, and 3 is the dimension of each point.
        eps: Small positive number to avoid division by zero in the computation of the torsion angle.

    Returns:
        torsion_angle: Tensor of torsion angles. Shape: (B, K-3).
        sign_torsion_angle: Tensor of signs of torsion angles. Shape: (B, K-3).
    """
    a, b, c = points[:, 0, :], points[:, 1, :], points[:, 2, :]
    d = points[:, 3:, :]

    # Compute the vectors between the first three points and all other points
    v1 = b.unsqueeze(1) - a.unsqueeze(1)
    v2 = c.unsqueeze(1) - b.unsqueeze(1)
    v3 = d - c.unsqueeze(1)

    n1 = torch.cross(v1, v2, dim=-1)
    n2 = torch.cross(v2, v3, dim=-1)

    # Normalize the normals
    n1 = n1 / (n1.norm(dim=-1, keepdim=True) + eps)
    n2 = n2 / (n2.norm(dim=-1, keepdim=True) + eps)

    # Compute the dot product and cross product between the two normals
    dot_product = torch.sum(n1 * n2, dim=-1)
    cross_product = torch.cross(n1, n2)

    # Calculate the torsion angle and its sign and save them for valid points
    torsion_angle = torch.atan2(torch.sum(v1 * cross_product, dim=-1), dot_product)
    sign_torsion_angle = torch.sign(torsion_angle)

    return torsion_angle, sign_torsion_angle




def create_torsion_mask(points, eps=1e-7, torsion_threshold=1e-3):
    """
    Create a mask that is True for all quadruplets of points that form a torsion angle higher than a certain threshold.

    Args:
        points: Input tensor. Shape: (B, K, 3), where B is the batch size, K is the number of points in each batch, and 3 is the dimension of each point.
        eps: Small positive number to avoid division by zero in the computation of the torsion angle.
        torsion_threshold: The threshold torsion angle for considering a set of points to be valid.

    Returns:
        torsion_mask: A tensor of the same shape as the input tensor, but with the second dimension reduced by 3. This mask has True for all points that form a valid torsion angle with the first three points.
    """
    torsion_angle, torsion_sign = compute_torsion_angles_ex(points, eps)
    abs_angle = torch.abs(torsion_angle)
    torsion_mask = ((abs_angle > torsion_threshold) & (abs_angle < np.pi - torsion_threshold))
    return torsion_mask


def combine_two_dims(tensor):
    """
    Combines two dimensions into one by creating composite keys
    Args:
        tensor: The tensor of shape (*, 2)
    Returns:
        keys: Tensor of composite keys
    """
    max_val = torch.max(tensor)  # Find the maximum value in the tensor
    keys = tensor[..., 0] * (max_val + 1) + tensor[..., 1]  # Create composite keys from two dimensions
    return keys


def combine_three_dims(tensor):
    """
    Combines three dimensions into one by creating composite keys
    Args:
        tensor: The tensor of shape (*, 3)
    Returns:
        keys: Tensor of composite keys
    """
    max_val = torch.max(tensor)  # Find the maximum value in the tensor
    max_weight = (max_val + 1)
    keys = tensor[..., 0] * max_weight * max_weight + tensor[..., 1] * max_weight + tensor[..., 2]  # Create composite keys from three dimensions
    return keys


def sort_knn_with_distances(knn_points, reference_points):
    """
    kNN by default sorted by distance, but we can have multiple reference points to break the ties.
    Args:
        knn_points (Tensor): BxKx3
        reference_points (Tensor): BxNx3, sorted as 1..N the last point is most significant.
        The rest of the points used as a tie braker in case previous points are equal.
    """
    knn_points_aligned = knn_points.view(knn_points.shape[0], -1, 1, 3)
    reference_points_aligned = reference_points.view(reference_points.shape[0], 1, -1, 3)
    # sort knn_points by distance to c_points
    # break the ties with angle between nn_points[:, 1:2, :] and c_points
    distances = torch.norm(knn_points_aligned - reference_points_aligned, dim=-1) # BxKxN
    # print(distances.shape[-1])
    if distances.shape[-1] >= 3:
        distances = distances[..., -3:].flip(-1)
        combined_props = combine_three_dims(distances) # BxKx1
    elif distances.shape[-1] == 2:
        distances = distances[..., -2:].flip(-1)
        combined_props = combine_two_dims(distances)
    else:
        combined_props = distances.squeeze(-1)
    # Sort 'sphere' along the second dimension and get the indices
    sorted_props, indices = torch.sort(combined_props.squeeze(-1), dim=1)
    # print(indices.shape, 'indices')
    # Unsqueeze and expand indices to match the dimensions of 'knn_points_masked'
    indices = indices.unsqueeze(-1).expand_as(knn_points)
    # Use these indices to sort 'knn_points_masked'
    sorted_knn_points_masked = knn_points.gather(1, indices)
    return sorted_knn_points_masked, indices, sorted_props



def compute_torsion_angles(points, mask=None, mask_value=0, eps=1e-7):
    a, b, c, d = points[:, 0, :], points[:, 1, :], points[:, 2, :], points[:, 3, :]
    v1 = b - a
    v2 = c - b
    v3 = d - c

    n1 = torch.cross(v1, v2, dim=-1)
    n2 = torch.cross(v2, v3, dim=-1)

    # Normalize the normals
    n1 = n1 / (n1.norm(dim=-1, keepdim=True) + eps)
    n2 = n2 / (n2.norm(dim=-1, keepdim=True) + eps)

    # Compute the dot product and cross product between the two normals
    dot_product = torch.sum(n1 * n2, dim=-1)
    cross_product = torch.cross(n1, n2)

    # Calculate the torsion angle and its sign and save them for valid points
    torsion_angle = torch.atan2(torch.sum(v1 * cross_product, dim=-1), dot_product)  # <-- This line was changed
    sign_torsion_angle = torch.sign(torsion_angle)

    # Apply the mask
    if mask is not None:
        torsion_angle[~mask.squeeze()] = mask_value
        sign_torsion_angle[~mask.squeeze()] = mask_value

    return torsion_angle, sign_torsion_angle


def create_unique_mask(tensor, eps=1e-5, sort=False):
    """
    Create a mask that is True for all elements that are unique in the tensor.

    This function works on tensors of any shape, but only considers uniqueness along the last dimension after reshaping to 2D.
    The uniqueness test is based on the sorted values and their adjacent elements, not on the original order of elements.

    Args:
        tensor: Input tensor. Shape: (N, M, ..., P) where N, M, ..., P are arbitrary positive integers.
                The tensor will be reshaped to 2D for processing, so its original shape doesn't matter.
        eps: Tolerance for the equality (closeness) check. Elements are considered equal if their difference is less than `eps`.
        sort: If True, the tensor will be sorted along the last dimension before checking for uniqueness.
        If False, the original order is expected to be sorted otherwise the result will be incorrect.

    Returns:
        all_same: A 2D mask tensor of the same shape as the reshaped input tensor, with True at positions of unique elements.
                  Shape: (N, M * ... * P)
    """
    # Reshape the tensor to 2D for processing
    tensor = tensor.view(tensor.shape[0], -1)

    if tensor.shape[1] == 1:
        # Base case: only one element. It's necessarily unique.
        return torch.ones_like(tensor, dtype=torch.bool)
    # Sort the tensor along the last dimension
    if sort:
        tensor, _ = torch.sort(tensor, dim=-1)

    # Compute a boolean mask that is True for unique elements
    # We check for uniqueness by comparing each element with its neighbors after sorting.
    # We use torch.isclose to compare for equality within a tolerance `eps`.
    all_same = ~torch.isclose(tensor, torch.roll(tensor, 1, dims=-1), atol=eps) & \
               ~torch.isclose(tensor, torch.roll(tensor, -1, dims=-1), atol=eps)
    return all_same


def split_point_with_idx(points, indices, idx, attr=None, pad_mask=None):
    """

    Args:
        points: BxKx3
        idx: B

    Returns:
        b_idx: Bx1x1
        b_point: Bx1x3

    """
    out_attr, remaining_attr  = None, None
    b_idx = indices[torch.arange(points.shape[0]), idx].unsqueeze(1)
    b_point = points[torch.arange(points.shape[0]), idx].view(points.shape[0], 1, points.shape[2])
    if attr is not None:
        out_attr = attr[torch.arange(points.shape[0]), idx].view(points.shape[0], 1)

    mask = torch.ones_like(points[:, :, 0], dtype=torch.bool)
    mask.scatter_(1, idx[:, None], 0)

    remaining = points[mask.view(mask.shape[0], mask.shape[1], 1).expand_as(points)].view(points.shape[0], -1, points.shape[2])
    if attr is not None:
        remaining_attr = attr[mask.view(mask.shape[0], mask.shape[1])].view(attr.shape[0], -1)

    if pad_mask is not None:
        pad_mask = pad_mask[mask.view(mask.shape[0], mask.shape[1])].view(pad_mask.shape[0], -1)
        return b_point, b_idx, remaining, out_attr, remaining_attr, pad_mask
    else:
        return b_point, b_idx, remaining, out_attr, remaining_attr



def calculate_stable_torsion(batch, k=5, add_self=True, distance_eps=2e-3, non_linear_eps=5e-5, torsion_eps=1e-3):
    if 'edge_index' not in batch:
        edge_index = knn_graph(batch.pos, k=k, batch=batch.batch, loop=False, )
    else:
        edge_index = batch.edge_index


    if add_self:
        edge_index2 = add_self_loops(edge_index, num_nodes=batch.num_nodes)

    j, i = edge_index2
    x_neigh, pad_mask = to_dense_batch(j, batch=i, fill_value=0) # here

    x_ids = torch.arange(batch.num_nodes, device=batch.pos.device).view(-1)
    knn_points = batch.pos[x_neigh]
    if knn_points.shape[1] < 4:
        return torch.zeros_like(x_ids), torch.zeros_like(x_ids).to(torch.bool), False
    x_points = x_ids[x_neigh]

    a_point = knn_points[:, 0:1, :]
    knn_points_new, indices, sorted_props = sort_knn_with_distances(knn_points[:, 1:, :], a_point)
    pad_mask = pad_mask[:, 1:].gather(1, indices[:, :, 0])
    sorted_x = x_points[:, 1:].gather(1, indices[:, :, 0])

    mask = create_unique_mask(sorted_props, eps=distance_eps)
    mask = mask & pad_mask
    first_valid = mask.to(torch.long).argmax(dim=1)
    is_valid_b = mask.sum(1) > 0
    b_point, b_idx, b_remaining, b_selected_x, bx_points, pad_mask = split_point_with_idx(knn_points_new, indices, first_valid, attr=sorted_x, pad_mask=pad_mask)

    knn_points_new, indices, sorted_props = sort_knn_with_distances(b_remaining, b_point)
    sorted_bx = bx_points.gather(1, indices[:, :, 0])

    mask = create_unique_mask(sorted_props, eps=distance_eps)
    linear_mask = create_non_linear_mask(torch.cat([a_point, b_point, knn_points_new], dim=1), linear_threshold=non_linear_eps)
    total_mask = linear_mask & mask & pad_mask

    first_valid = total_mask.to(torch.long).argmax(dim=1)
    is_valid_c= total_mask.sum(1) > 0

    c_point, c_idx, c_remaining, c_selected_x, cx_points, pad_mask = split_point_with_idx(knn_points_new, indices, first_valid, attr=sorted_bx, pad_mask=pad_mask)


    knn_points_new, indices, sorted_props = sort_knn_with_distances(c_remaining, c_point)

    sorted_cx = cx_points.gather(1, indices[:, :, 0])
    mask = create_unique_mask(sorted_props, eps=distance_eps)
    linear_mask = create_non_linear_mask(torch.cat([b_point, c_point, knn_points_new], dim=1), linear_threshold=non_linear_eps)
    torsion_mask = create_torsion_mask(torch.cat([a_point, b_point, c_point, knn_points_new], dim=1), torsion_threshold=torsion_eps)


    total_mask = linear_mask & mask & torsion_mask & pad_mask
    first_valid = total_mask.to(torch.long).argmax(dim=1)
    is_valid_d= total_mask.sum(1) > 0

    d_point, d_idx, d_remaining, d_selected_x, dx_points  = split_point_with_idx(knn_points_new, indices, first_valid, attr=sorted_cx)
    all_valid = is_valid_b & is_valid_c & is_valid_d

    selected = torch.cat([knn_points[:, 0:1, :], b_point, c_point, d_point], dim=1)
    selected_idx = torch.cat([x_points[:, 0:1], b_selected_x, c_selected_x, d_selected_x], dim=1)

    return selected, all_valid, selected_idx


def stable_torsion_wrapper(batch, k=6, torsion_type="simple"):
    if torsion_type == "stable":
        points, mask, selected = calculate_stable_torsion(batch, k=k)
    else:
        raise NotImplementedError(f"torsion type {torsion_type} not implemented")
    if selected is False:
        return torch.zeros_like(points), torch.zeros_like(points).to(torch.long), False
    tors, sign  = compute_torsion_angles(points, mask, mask_value=0, eps=1e-7)
    return tors, sign, selected

