"""FDE Tidy3d backend (default backend for MEOW)."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from types import SimpleNamespace
from typing import Literal

import numpy as np
from pydantic import PositiveFloat, PositiveInt
from scipy.constants import c
from tidy3d.components.mode.solver import compute_modes as _compute_modes

from meow.cross_section import CrossSection
from meow.fde.post_process import post_process_modes
from meow.mode import Mode, Modes, inner_product, normalize

def compute_modes_tidy3d(
    cs: CrossSection,
    num_modes: PositiveInt = 10,
    target_neff: PositiveFloat | None = None,
    precision: Literal["single", "double"] = "double",
    post_process: Callable = post_process_modes,
) -> Modes:
    """Compute ``Modes`` for a given ``CrossSection``.

    Args:
        cs: the cross-section to solve modes for.
        num_modes: number of modes to compute.
        target_neff: effective index near which to search for modes.
        precision: floating-point precision, ``"single"`` or ``"double"``.
        post_process: callable applied to the raw mode list before returning.

    Returns:
        The computed and post-processed collection of modes.
    """
    if num_modes < 1:
        msg = "You need to request at least 1 mode."
        raise ValueError(msg)

    od = np.zeros_like(cs.nx)  # off diagonal entry
    eps_cross = [cs.nx**2, od, od, od, cs.ny**2, od, od, od, cs.nz**2]

    if np.isinf(cs.mesh.bend_radius) or np.isnan(cs.mesh.bend_radius):
        bend_radius = None
        bend_axis = None
    else:
        bend_radius = cs.mesh.bend_radius
        bend_axis = cs.mesh.bend_axis

    mode_spec = SimpleNamespace(  # tidy3d.ModeSpec alternative (prevents type checking)
        num_modes=num_modes,
        target_neff=target_neff,
        num_pml=cs.mesh.num_pml,
        filter_pol=None,
        angle_theta=cs.mesh.angle_theta,
        angle_phi=cs.mesh.angle_phi,
        bend_radius=bend_radius,
        precision=precision,
        bend_axis=bend_axis,
        track_freq="central",
        group_index_step=False,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Input has data type int64.*")
        warnings.filterwarnings("ignore", message=".*divide by zero.*")
        warnings.filterwarnings("ignore", message=".*overflow encountered.*")
        warnings.filterwarnings("ignore", message=".*invalid value.*")
        ((Ex, Ey, Ez), (Hx, Hy, Hz)), neffs = (
            x.squeeze()
            for x in _compute_modes(
                eps_cross=eps_cross,
                coords=[cs.mesh.x, cs.mesh.y],
                freq=c / (cs.env.wl * 1e-6),
                mode_spec=mode_spec,
                precision=precision,
                plane_center=cs.mesh.plane_center,
            )[:2]
        )

    if num_modes == 1:
        modes = [
            Mode(
                cs=cs,
                Ex=Ex,
                Ey=Ey,
                Ez=Ez,
                Hx=Hx,
                Hy=Hy,
                Hz=Hz,
                neff=np.asarray(neffs, dtype=np.complex128).item(),
            )
            for _ in range(num_modes)
        ]
    else:  # num_modes > 1
        modes = [
            Mode(
                cs=cs,
                Ex=Ex[..., i],
                Ey=Ey[..., i],
                Ez=Ez[..., i],
                Hx=Hx[..., i],
                Hy=Hy[..., i],
                Hz=Hz[..., i],
                neff=neffs[i],
            )
            for i in range(num_modes)
        ]

    modes = sorted(modes, key=lambda m: float(np.real(m.neff)), reverse=True)

    # Normalize the modes with respect to the inner product, which is important for accurate mode overlap calculations.
    modes1 = []
    for mode in modes:
        modes1.append(normalize(mode, inner_product))
    return modes1 # post_process(modes) BAD: this orthonormalizes the modes with respect to the L2 inner product, which for whatever reason causes the effective indices to change which breaks the eme solver
