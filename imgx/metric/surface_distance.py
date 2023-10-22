"""Surface distance metric.

Functions are all numpy based, as they rely on scipy and not jittable for JAX.

References:
    https://github.com/deepmind/surface-distance
    https://github.com/Project-MONAI/MONAI/blob/dev/monai/metrics/surface_distance.py
"""
from __future__ import annotations

from functools import partial
from typing import Callable

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

from imgx_datasets.preprocess import get_binary_mask_bounding_box


def get_mask_edges(mask_pred: np.ndarray, mask_true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Do binary erosion and use XOR for input to get the edges.

    Args:
        mask_pred: the predicted binary mask.
        mask_true: the ground truth binary mask.

    Returns:
        edge_pred: the predicted binary edge.
        edge_true: the ground truth binary edge.
    """
    if mask_pred.dtype != np.bool_:
        mask_pred = mask_pred > 0
    if mask_true.dtype != np.bool_:
        mask_true = mask_true > 0

    mask_union = mask_pred | mask_true
    if not mask_union.any():
        # if no foreground prediction and ground truth, return zero
        return np.zeros_like(mask_union), np.zeros_like(mask_union)

    bbox_min, bbox_max = get_binary_mask_bounding_box(mask=mask_union)
    for i, (v_min, v_max) in enumerate(zip(bbox_min, bbox_max)):
        mask_pred = mask_pred.take(indices=range(v_min, v_max), axis=i)
        mask_true = mask_true.take(indices=range(v_min, v_max), axis=i)
    edge_pred = binary_erosion(mask_pred) ^ mask_pred
    edge_true = binary_erosion(mask_true) ^ mask_true
    return edge_pred, edge_true


def get_surface_distance(
    edge_pred: np.ndarray,
    edge_true: np.ndarray,
    spacing: tuple[float, ...] | None = None,
) -> np.ndarray:
    """Calculate surface distance from predicted edges to ground truth.

    Args:
        edge_pred: the predicted binary edge.
        edge_true: the ground truth binary edge.
        spacing: spacing of pixel/voxels along each dimension.

    Returns:
        surface distance, 1D array of len = edge size.
    """
    distance_transform = distance_transform_edt(input=~edge_true, sampling=spacing)
    surface_distance = np.asarray(distance_transform[edge_pred])
    return surface_distance


def _aggregated_symmetric_surface_distance(
    dist_pred_true: np.ndarray,
    dist_true_pred: np.ndarray,
    f: Callable,
    num_args: int,
) -> float:
    """Aggregate surface distance in a symmetric way.

    Args:
        dist_pred_true: surface distance from predicted edges to ground truth.
        dist_true_pred: surface distance from ground truth edges to predicted.
        f: an aggregation function taking one or two arguments.
        num_args: number of arguments for f. It has to be passed manually,
            because it is not possible to get the number for partial functions.

    Returns:
        Aggregated values.
    """
    if num_args == 1:
        return max(f(dist_pred_true), f(dist_true_pred))
    if num_args == 2:
        return f(dist_pred_true, dist_true_pred)
    raise ValueError(
        "Symmetric surface distance aggregation function "
        f"should take one or two arguments, got {num_args}."
    )


def _aggregated_surface_distance(
    mask_pred: np.ndarray,
    mask_true: np.ndarray,
    agg_fn_list: list[Callable],
    num_args_list: list[int],
    spacing: tuple[float, ...] | None,
    symmetric: bool = True,
) -> np.ndarray:
    """Calculate aggregated surface distance.

    Args:
        mask_pred: one hot predictions with only spatial dimensions.
        mask_true: one hot targets with only spatial dimensions.
        agg_fn_list: a list of functions
            to aggregate a list of distances.
        num_args_list: a list of ints, corresponding to number of arguments
            for agg_fn_list.
        spacing: spacing of pixel/voxels along each dimension.
        symmetric: the distance is symmetric to (pred, true) means swapping
            the masks provides the same value.

    Returns:
        Aggregated surface distance of shape (batch, num_classes).
    """
    if not (mask_pred.any() and mask_true.any()):
        # prediction or ground truth do not have foreground
        return np.array([np.nan for _ in agg_fn_list])
    edge_pred, edge_true = get_mask_edges(
        mask_pred=mask_pred,
        mask_true=mask_true,
    )
    dist_pred_true = get_surface_distance(edge_pred=edge_pred, edge_true=edge_true, spacing=spacing)
    if symmetric:
        # swap edge_pred, edge_true and calculate again
        dist_true_pred = get_surface_distance(
            edge_pred=edge_true, edge_true=edge_pred, spacing=spacing
        )
        if dist_pred_true.size == 0 or dist_true_pred.size == 0:
            return np.array([np.nan for _ in agg_fn_list])
        return np.array(
            [
                _aggregated_symmetric_surface_distance(
                    dist_pred_true=dist_pred_true,
                    dist_true_pred=dist_true_pred,
                    f=f,
                    num_args=num_args,
                )
                for f, num_args in zip(agg_fn_list, num_args_list)
            ]
        )
    # not symmetric, just need dist_pred_true
    if dist_pred_true.size == 0:
        return np.array([np.nan for _ in agg_fn_list])
    return np.array([f(dist_pred_true) for f in agg_fn_list])


def aggregated_surface_distance(
    mask_pred: np.ndarray,
    mask_true: np.ndarray,
    agg_fns: Callable | list[Callable],
    num_args: int | list[int],
    spacing: tuple[float, ...] | None,
    symmetric: bool = True,
) -> np.ndarray:
    """Calculate aggregated surface distance on batch.

    Args:
        mask_pred: one hot predictions, (batch, ..., num_classes).
        mask_true: one hot targets, (batch, ..., num_classes).
        agg_fns: a function or a list of functions
            to aggregate a list of distances.
        num_args: an int or a list of ints, corresponding to number of arguments
            for agg_fn_list.
        spacing: spacing of pixel/voxels along each dimension.
        symmetric: the distance is symmetric to (pred, true) means swapping
            the masks provides the same value.

    Returns:
        Aggregated surface distance of shape (num_funcs, batch, num_classes).
    """
    agg_fn_list = agg_fns if isinstance(agg_fns, list) else [agg_fns]
    num_args_list = num_args if isinstance(num_args, list) else [num_args]
    if len(agg_fn_list) != len(num_args_list):
        raise ValueError("agg_funcs and num_args lengths do not match.")

    num_agg_fns = len(agg_fn_list)
    batch_size = mask_pred.shape[0]
    num_classes = mask_pred.shape[-1]
    agg_dist_arr = np.zeros((num_agg_fns, batch_size, num_classes))
    for batch_index, class_idx in np.ndindex(batch_size, num_classes):
        agg_dist_arr[:, batch_index, class_idx] = _aggregated_surface_distance(
            mask_pred=mask_pred[batch_index, ..., class_idx],
            mask_true=mask_true[batch_index, ..., class_idx],
            agg_fn_list=agg_fn_list,
            num_args_list=num_args_list,
            spacing=spacing,
            symmetric=symmetric,
        )
    if num_agg_fns == 1:
        return agg_dist_arr[0, ...]
    return agg_dist_arr


def average_surface_distance(
    mask_pred: np.ndarray,
    mask_true: np.ndarray,
    spacing: tuple[float, ...] | None,
    symmetric: bool = True,
) -> np.ndarray:
    """Calculate average surface distance on batch.

    Args:
        mask_pred: one hot predictions, (batch, ..., num_classes).
        mask_true: one hot targets, (batch, ..., num_classes).
        spacing: spacing of pixel/voxels along each dimension.
        symmetric: the distance is symmetric to (pred, true) means swapping
            the masks provides the same value.

    Returns:
        Average surface distance of shape (batch, num_classes).
    """
    return aggregated_surface_distance(
        mask_pred=mask_pred,
        mask_true=mask_true,
        agg_fns=np.mean,
        num_args=1,
        spacing=spacing,
        symmetric=symmetric,
    )


def hausdorff_distance(
    mask_pred: np.ndarray,
    mask_true: np.ndarray,
    percentile: int,
    spacing: tuple[float, ...] | None,
    symmetric: bool = True,
) -> np.ndarray:
    """Calculate hausdorff distance on batch.

    Args:
        mask_pred: one hot predictions, (batch, ..., num_classes).
        mask_true: one hot targets, (batch, ..., num_classes).
        percentile: hausdorff distance is the percentile of surface distances.
        spacing: spacing of pixel/voxels along each dimension.
        symmetric: the distance is symmetric to (pred, true) means swapping
            the masks provides the same value.

    Returns:
        Hausdorff distance of shape (batch, num_classes).
    """
    return aggregated_surface_distance(
        mask_pred=mask_pred,
        mask_true=mask_true,
        agg_fns=partial(np.percentile, q=percentile),
        num_args=1,
        spacing=spacing,
        symmetric=symmetric,
    )


def normalized_surface_dice_from_distances(
    dist_pred_true: np.ndarray,
    dist_true_pred: np.ndarray,
    tolerance_mm: float = 1.0,
) -> float:
    """Calculate the normalized surface dice at a specified tolerance.

    The implementation is similar to MONAI
    https://github.com/Project-MONAI/MONAI/blob/dev/monai/metrics/surface_dice.py,
    and different from DeepMind,
    https://github.com/deepmind/surface-distance/blob/master/surface_distance/metrics.py,
    where deepmind used Marching Cubes,
    https://graphics.stanford.edu/~mdfisher/MarchingCubes.html
    to estimate surface area corresponding to each voxel,
    and is therefore more accurate but slower.
    MONAI's implementation uses surface distances only,
    and per voxel on the boundary/edge has equal weights, unlike DeepMind.

    Args:
        dist_pred_true: surface distance from predicted edges to ground truth.
        dist_true_pred: surface distance from ground truth edges to predicted.
        tolerance_mm: tolerance value to consider surface being overlapping.

    Returns:
        A float value between [0.0, 1.0].
    """
    boundary_complete = len(dist_pred_true) + len(dist_true_pred)

    if boundary_complete == 0:
        # the class is neither present in the prediction
        # nor in the reference segmentation
        return np.nan
    boundary_correct = np.sum(dist_pred_true <= tolerance_mm) + np.sum(
        dist_true_pred <= tolerance_mm
    )
    return boundary_correct / boundary_complete


def normalized_surface_dice(
    mask_pred: np.ndarray,
    mask_true: np.ndarray,
    spacing: tuple[float, ...] | None,
    tolerance_mm: float = 1.0,
) -> np.ndarray:
    """Calculate the normalized surface dice at a specified tolerance on batch.

    Args:
        mask_pred: one hot predictions, (batch, ..., num_classes).
        mask_true: one hot targets, (batch, ..., num_classes).
        spacing: spacing of pixel/voxels along each dimension.
        tolerance_mm: tolerance value to consider surface being overlapping.

    Returns:
        Hausdorff distance of shape (batch, num_classes).
    """
    return aggregated_surface_distance(
        mask_pred=mask_pred,
        mask_true=mask_true,
        agg_fns=partial(normalized_surface_dice_from_distances, tolerance_mm=tolerance_mm),
        num_args=2,
        spacing=spacing,
        symmetric=True,
    )
