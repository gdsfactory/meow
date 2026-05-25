"""FDE Implementations & Backends."""

from __future__ import annotations

from meow.fde.default import (
    compute_modes,
)
from meow.fde.lumerical import (
    Sim,
    compute_modes_lumerical,
    create_lumerical_geometries,
    get_sim,
)
from meow.fde.post_process import (
    filter_modes,
    normalize_modes,
    orthonormalize_modes,
    post_process_modes,
)
from meow.fde.tidy3d import (
    compute_modes_tidy3d,
)
from meow.fde.meep import (
    compute_modes_meep,
)

__all__ = [
    "Sim",
    "compute_modes",
    "compute_modes_lumerical",
    "compute_modes_tidy3d",
    "create_lumerical_geometries",
    "filter_modes",
    "get_sim",
    "normalize_modes",
    "orthonormalize_modes",
    "post_process_modes",
]
