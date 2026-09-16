"""Mode post-processing utilities."""

from collections.abc import Callable, Iterable

import numpy as np

from meow.mode import (
    Modes,
    inner_product,
    is_lossy_mode,
    is_pml_mode,
    normalize,
    zero_phase,
)


def post_process_modes(
    modes: Modes,
    inner_product: Callable = inner_product,
    gm_tolerance: float = 0.01,
    neff_tolerance: float = 1e-6,
) -> Modes:
    """Default post-processing pipeline after FDE.

    Args:
        modes: the modes to post process
        inner_product: the inner product with which to post-process the modes
        gm_tolerance: Gram-Schmidt drop tolerance.
        neff_tolerance: maximum absolute effective-index difference for two
            modes to be treated as degenerate.

    Returns:
        Filtered and normalized modes with degenerate eigenspaces
        orthonormalized.

    Notes:
        This is the default ``post_process`` used by ``compute_modes``. The
        choice of ``inner_product`` here matters downstream: interface overlaps
        should be built with the same inner product.
    """
    return orthonormalize_modes(
        filter_modes(modes),
        inner_product,
        tolerance=gm_tolerance,
        neff_tolerance=neff_tolerance,
    )


def filter_modes(
    modes: Modes,
    conditions: Iterable[Callable] = (is_pml_mode, is_lossy_mode),
) -> Modes:
    """Filter a set of modes according to certain criteria.

    Args:
        modes: the list of modes to filter
        conditions: the conditions to filter the modes with

    Returns:
        the filtered modes
    """
    kept = []
    for mode in modes:
        for condition_fn in conditions:
            if condition_fn(mode):
                break
        else:
            kept.append(mode)
    return kept


def normalize_modes(modes: Modes, inner_product: Callable) -> Modes:
    """Self-normalize a set of modes.

    This only fixes the norm of each mode individually. Distinct eigenmodes are
    expected to be orthogonal under the appropriate modal inner product;
    :func:`orthonormalize_modes` additionally orthogonalizes degenerate
    eigenspaces.

    Args:
        modes: the modes to normalize.
        inner_product: callable computing the inner product between two modes.

    Returns:
        The normalized (and zero-phased) modes.
    """
    return [zero_phase(normalize(m, inner_product)) for m in modes]


def orthonormalize_modes(
    modes: Modes,
    inner_product: Callable,
    *,
    tolerance: float = 0.01,
    neff_tolerance: float = 1e-6,
) -> Modes:
    """Normalize modes and orthogonalize degenerate eigenspaces.

    Args:
        modes: the modes to orthonormalize
        inner_product: the inner product to orthonormalize them under
        tolerance: any mode that can't expand the basis beyond this tolerance
            will be dropped.
        neff_tolerance: maximum absolute effective-index difference for two
            modes to be treated as degenerate.

    Returns:
        Normalized mode basis with degenerate eigenspaces orthonormalized.

    Notes:
        Gram-Schmidt is only valid within a degenerate eigenspace. A linear
        combination of modes with different effective indices is not itself an
        eigenmode and has no single effective index. The ``inner_product``
        passed here should generally be the same one used later to build
        interface overlaps.
    """
    if not modes:
        return []
    modes = normalize_modes(modes, inner_product)
    basis = []
    for mode in modes:
        current = mode
        for b in basis:
            if abs(mode.neff - b.neff) > neff_tolerance:
                continue
            current = current - (inner_product(b, current) / inner_product(b, b)) * b
            # Mode arithmetic combines the fields and effective indices. Within
            # a degenerate eigenspace the fields should be combined while the
            # eigenvalue of the original mode is retained.
            current = current.model_copy(update={"neff": mode.neff})
        norm_sq = inner_product(current, current)
        if abs(norm_sq) < tolerance:
            continue
        normalized = current / np.sqrt(norm_sq)
        basis.append(normalized.model_copy(update={"neff": mode.neff}))
    return basis
