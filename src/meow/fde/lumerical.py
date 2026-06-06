"""FDE Lumerical Backend."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

import numpy as np
from pydantic import PositiveInt

from meow.arrays import ComplexArray2D
from meow.cross_section import CrossSection
from meow.environment import Environment
from meow.fde.post_process import post_process_modes
from meow.mode import Mode
from meow.structures import Structure3D

Sim: TypeAlias = Any
"""A lumerical simulation object."""
_sim: Sim | None = None


def compute_modes_lumerical(
    cs: CrossSection,
    num_modes: PositiveInt = 10,
    unit: float = 1e-6,
    post_process: Callable = post_process_modes,
    sim: Sim | None = None,
) -> list[Mode]:
    """Compute ``Modes`` for a given ``CrossSection`` (Lumerical backend).

    Args:
        cs: the cross-section to solve modes for.
        num_modes: number of modes to compute.
        unit: unit scaling factor (default 1e-6 for micrometres).
        post_process: callable applied to the raw mode list before returning.
        sim: optional Lumerical simulation object; uses the global one if None.

    Returns:
        The computed and post-processed list of modes.
    """
    from lumapi import (  # ty: ignore[unresolved-import]
        LumApiError,  # type: ignore[reportMissingImports]
    )

    sim = get_sim(sim=sim)

    if sim is None:
        msg = "Lumerical simulation object not given or found."
        raise RuntimeError(msg)

    cell = cs._cell

    if cell is None:
        msg = (
            "The cross-section does not have a cell defined. "
            "Please define a cell before computing modes."
        )
        raise ValueError(msg)

    _assert_default_mesh_setting(cell.mesh.angle_phi == 0, "angle_phi")
    _assert_default_mesh_setting(cell.mesh.angle_theta == 0, "angle_theta")
    _assert_default_mesh_setting(cell.mesh.bend_radius is None, "bend_radius")

    create_lumerical_geometries(sim, cell.structures, cs.env, unit)

    sim.select("FDE")
    sim.delete()
    pml_settings = {}
    num_pml_y, num_pml_z = 0, 0
    if cell.mesh.num_pml[0] > 0:
        pml_settings.update(
            {
                "y_min_bc": "PML",
                "y_max_bc": "PML",
            }
        )
        num_pml_y = 22  # TODO: allow adjusting these values
    if cell.mesh.num_pml[1] > 0:
        pml_settings.update(
            {
                "z_min_bc": "PML",
                "z_max_bc": "PML",
            }
        )
        num_pml_z = 22  # TODO: allow adjusting these values
    sim.addfde(
        background_index=1.0,
        solver_type="2D X normal",
        x=float(cell.z * unit),
        y_min=float(cell.mesh.x.min() * unit),
        y_max=float(cell.mesh.x.max() * unit),
        z_min=float(cell.mesh.y.min() * unit),
        z_max=float(cell.mesh.y.max() * unit),
        define_y_mesh_by="number of mesh cells",
        define_z_mesh_by="number of mesh cells",
        mesh_cells_y=cell.mesh.x_.shape[0],
        mesh_cells_z=cell.mesh.y_.shape[0],
        **pml_settings,
    )
    # set mesh size again, because PML messes with it:
    if cell.mesh.num_pml[0] > 0:
        sim.setnamed("FDE", "mesh cells y", cell.mesh.x_.shape[0] - num_pml_y)
    if cell.mesh.num_pml[1] > 0:
        sim.setnamed("FDE", "mesh cells z", cell.mesh.y_.shape[0] - num_pml_z)
    sim.setanalysis("number of trial modes", int(num_modes))
    sim.setanalysis("search", "near n")
    sim.setanalysis("use max index", True)  # noqa: FBT003
    sim.setanalysis("wavelength", float(cs.env.wl * unit))
    sim.findmodes()
    modes = []
    for j in range(1, num_modes + 1):
        try:
            mode = _lumerical_fields_to_mode(
                cs=cs,
                lneff=sim.getdata(f"mode{j}", "neff"),
                lEx=sim.getdata(f"mode{j}", "Ex"),
                lEy=sim.getdata(f"mode{j}", "Ey"),
                lEz=sim.getdata(f"mode{j}", "Ez"),
                lHx=sim.getdata(f"mode{j}", "Hx"),
                lHy=sim.getdata(f"mode{j}", "Hy"),
                lHz=sim.getdata(f"mode{j}", "Hz"),
            )
        except LumApiError:
            break
        modes.append(mode)

    modes = sorted(modes, key=lambda m: np.real(m.neff), reverse=True)
    return post_process(modes)


def create_lumerical_geometries(
    sim: Sim,
    structures: list[Structure3D],
    env: Environment,
    unit: float,
) -> None:
    """Create Lumerical geometries from a list of structures.

    Args:
        sim: the Lumerical simulation object.
        structures: 3-D structures to add to the simulation.
        env: the environment containing material/wavelength info.
        unit: unit scaling factor.
    """
    sim = get_sim(sim=sim)
    sim.switchtolayout()
    sim.deleteall()
    for s in structures:
        s._lumadd(sim, env, unit, "yzx")


def get_sim(**kwargs: Any) -> Sim:
    """Get the Lumerical simulation object.

    Args:
        **kwargs: keyword arguments; pass ``sim`` to set and return a specific
            simulation object.

    Returns:
        The active Lumerical simulation object.
    """
    global _sim  # noqa: PLW0603
    sim = kwargs.get("sim", None)
    if sim is not None:
        _sim = sim
        return sim
    sim = _sim
    if sim is None:
        msg = (
            "Could not start Lumerical simulation. Please either pass the "
            "`lumapi.MODE` simulation object as an argument to "
            "`compute_modes_lumerical` to globally "
            "set the lumapi.MODE simulation object."
        )
        raise ValueError(msg)
    return sim


def _lumerical_fields_to_mode(
    cs: CrossSection,
    lneff: ComplexArray2D,
    lEx: ComplexArray2D,
    lEy: ComplexArray2D,
    lEz: ComplexArray2D,
    lHx: ComplexArray2D,
    lHy: ComplexArray2D,
    lHz: ComplexArray2D,
) -> Mode:
    return Mode(
        cs=cs,
        neff=lneff.ravel().item(),
        Ex=lEy.squeeze()[1:, :-1],
        Ey=lEz.squeeze()[:-1, 1:],
        Ez=lEx.squeeze()[:-1, :-1],
        Hx=lHy.squeeze()[:-1, 1:],
        Hy=lHz.squeeze()[1:, :-1],
        Hz=lHx.squeeze()[1:, 1:],
    )


def _assert_default_mesh_setting(condition: bool, param_name: str) -> None:  # noqa: FBT001
    if not condition:
        msg = (
            f"Setting mesh.{param_name} is currently not supported in the "
            "Lumerical Backend. Please open an issue of submit a PR on GitHub "
            "to fix this: https://github.com/gdsfactory/meow",
        )
        raise NotImplementedError(msg)
