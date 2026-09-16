"""FDE EigenLight backend."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
from pydantic import PositiveFloat, PositiveInt

from meow.cross_section import CrossSection
from meow.fde.post_process import post_process_modes
from meow.mode import Mode, Modes


def compute_modes_eigenlight(
    cs: CrossSection,
    num_modes: PositiveInt = 10,
    target_neff: PositiveFloat | None = None,
    precision: Literal["single", "double"] = "double",
    post_process: Callable = post_process_modes,
) -> Modes:
    """Compute modes using the EigenLight FDFD solver.

    Args:
        cs: Cross-section to solve modes for.
        num_modes: Number of modes to compute.
        target_neff: Effective index around which EigenLight applies its
            shift-invert eigensolve. Defaults to the largest real refractive
            index in the cross-section.
        precision: Floating-point precision. EigenLight currently supports
            double precision only.
        post_process: Callable applied to the raw mode list before returning.

    Returns:
        The computed and post-processed modes.

    Raises:
        ModuleNotFoundError: If the optional EigenLight package is not installed.
        NotImplementedError: If single precision or a y-axis bend is requested.
    """
    if num_modes < 1:
        msg = "You need to request at least 1 mode."
        raise ValueError(msg)
    if precision != "double":
        msg = "compute_modes_eigenlight does not yet support precision != 'double'."
        raise NotImplementedError(msg)

    bend_radius = _bend_radius(cs)
    if bend_radius is not None and cs.mesh.bend_axis != 0:
        msg = "EigenLight currently supports bends along the x-axis only."
        raise NotImplementedError(msg)

    try:
        import eigenlight  # ty: ignore[unresolved-import]
    except ModuleNotFoundError as error:
        msg = (
            "The EigenLight backend requires the optional 'eigenlight' package. "
            "Install it with `pip install meow-sim[eigenlight]` under CPython "
            "3.12, the version supported by EigenLight 0.0.1's PyPI wheels."
        )
        raise ModuleNotFoundError(msg) from error

    target = float(target_neff) if target_neff is not None else _target_neff(cs)
    result = eigenlight.compute_modes(
        eps_xx=np.ascontiguousarray(cs.nx**2, dtype=np.complex128),
        eps_yy=np.ascontiguousarray(cs.ny**2, dtype=np.complex128),
        eps_zz=np.ascontiguousarray(cs.nz**2, dtype=np.complex128),
        x=np.ascontiguousarray(cs.mesh.x, dtype=np.float64),
        y=np.ascontiguousarray(cs.mesh.y, dtype=np.float64),
        wavelength=float(cs.env.wl),
        num_modes=int(num_modes),
        target_neff=target,
        num_pml=tuple(int(n) for n in cs.mesh.num_pml),
        bend_radius=bend_radius,
        angle_theta=float(cs.mesh.angle_theta),
        angle_phi=float(cs.mesh.angle_phi),
    )

    neffs = np.asarray(result.neff) + 1j * np.asarray(result.keff)
    field_arrays = {
        name: np.asarray(getattr(result, name.lower()), dtype=np.complex128)
        for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    }
    modes = [
        Mode(
            cs=cs,
            neff=neff,
            **{name: fields[i] for name, fields in field_arrays.items()},
        )
        for i, neff in enumerate(neffs)
    ]
    modes.sort(key=lambda mode: float(np.real(mode.neff)), reverse=True)
    return post_process(modes)


def _target_neff(cs: CrossSection) -> float:
    """Choose a shift near the highest-index material for the default solve."""
    return float(
        max(
            np.max(np.real(cs.nx)),
            np.max(np.real(cs.ny)),
            np.max(np.real(cs.nz)),
        )
    )


def _bend_radius(cs: CrossSection) -> float | None:
    """Translate MEOW's unset bend-radius values to EigenLight's ``None``."""
    bend_radius = float(cs.mesh.bend_radius)
    return None if not np.isfinite(bend_radius) else bend_radius
