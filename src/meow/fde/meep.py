"""FDE MEEP backend."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from pydantic import PositiveFloat, PositiveInt

from meow.cross_section import CrossSection
from meow.environment import Environment
from meow.fde.post_process import post_process_modes
from meow.geometries import Polygon2D, Rectangle
from meow.materials import Material
from meow.mode import Mode, Modes
from meow.structures import Structure2D, sort_structures


def compute_modes_meep(
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
    if target_neff is not None:
        msg = "compute_modes_meep does not yet support target_neff."
        raise NotImplementedError(msg)
    if precision != "double":
        msg = "compute_modes_meep does not yet support precision != 'double'."
        raise NotImplementedError(msg)

    # The mode-solve grid must match the CrossSection's own declared mesh
    # (cs.mesh.x / cs.mesh.y) rather than the raw bounding box of the
    # extruded structures: cladding layers routinely extend past the
    # intended mesh window (e.g. a larger vertical extrusion span than the
    # mesh's own y-extent), and deriving Nx/Ny from that mismatched bbox
    # produces a field grid whose shape doesn't match cs.mesh — breaking
    # inner_product()/normalize() downstream. mp.Block geometries are still
    # built from each structure's own true extent (a block wider than the
    # simulation cell is simply clipped by meep, which is correct).
    x_steps = np.diff(cs.mesh.x)
    y_steps = np.diff(cs.mesh.y)
    dx = float(x_steps[0])
    dy = float(y_steps[0])
    if (
        dx <= 0
        or dy <= 0
        or not np.allclose(x_steps, dx)
        or not np.allclose(y_steps, dy)
        or not np.isclose(dx, dy)
    ):
        msg = (
            "compute_modes_meep requires a uniform square mesh with positive, "
            f"equal x/y spacing; got dx={x_steps}, dy={y_steps}."
        )
        raise ValueError(msg)

    import meep as mp  # ty: ignore[unresolved-import]

    mp.verbosity(0)  # Suppress MEEP output

    # Later MEEP objects take precedence. MEOW's descending mesh-order sort
    # therefore gives lower mesh-order structures the same winning priority as
    # its rasterizer, without making assumptions based on refractive index.
    geometry = [
        _meep_geometry(struct, cs.env, mp) for struct in sort_structures(cs.structures)
    ]
    # mode fields must live on the mesh's cell-centered grid (mesh.x_/y_,
    # length N-1) to match what inner_product()/normalize() expect — not
    # the N-point vertex grid (mesh.x/y).
    Nx = len(cs.mesh.x_)
    Ny = len(cs.mesh.y_)
    x_span = cs.mesh.x[-1] - cs.mesh.x[0]
    y_span = cs.mesh.y[-1] - cs.mesh.y[0]
    x_center = (cs.mesh.x[-1] + cs.mesh.x[0]) / 2
    y_center = (cs.mesh.y[-1] + cs.mesh.y[0]) / 2

    sim = mp.Simulation(
        cell_size=mp.Vector3(x_span, y_span, 1),
        geometry_center=mp.Vector3(x_center, y_center),
        geometry=geometry,
        eps_averaging=True,
        resolution=1 / dx,
    )
    geometry_lattice = mp.Volume(
        center=(mp.Vector3(x_center, y_center, 0)), size=(mp.Vector3(x_span, y_span, 0))
    )
    try:
        sim.init_sim()

        modes = []
        for mode_num in range(1, num_modes + 1):
            mode_data = sim.get_eigenmode(
                frequency=1 / cs.env.wl,
                direction=mp.NO_DIRECTION,
                where=geometry_lattice,
                band_num=mode_num,
                parity=mp.NO_PARITY,
                kpoint=mp.Vector3(z=1),
                eigensolver_tol=1e-12,
            )
            y = cs.mesh.y_
            x = cs.mesh.x_
            components = {
                "Ex": mp.Ex,
                "Ey": mp.Ey,
                "Ez": mp.Ez,
                "Hx": mp.Hx,
                "Hy": mp.Hy,
                "Hz": mp.Hz,
            }
            fields = {
                name: np.zeros((Nx, Ny), dtype=np.complex128) for name in components
            }
            for i in range(Nx):
                for j in range(Ny):
                    point = mp.Vector3(x[i], y[j])
                    for name, component in components.items():
                        fields[name][i, j] = mode_data.amplitude(
                            point=point, component=component
                        )
            neff = mode_data.k[2] * cs.env.wl
            modes.append(Mode(cs=cs, neff=neff, **fields))

        modes = sorted(modes, key=lambda m: float(np.real(m.neff)), reverse=True)
        return post_process(modes)
    finally:
        # Free this simulation even if the eigensolver or post-processing fails.
        sim.reset_meep()


def _meep_geometry(structure: Structure2D, env: Environment, mp: Any) -> Any:
    """Convert a MEOW 2D structure to a MEEP geometric object."""
    _n, material = meep_material(structure.material, env)
    geometry = structure.geometry
    if isinstance(geometry, Rectangle):
        return mp.Block(
            size=mp.Vector3(
                geometry.x_max - geometry.x_min,
                geometry.y_max - geometry.y_min,
                mp.inf,
            ),
            center=mp.Vector3(
                0.5 * (geometry.x_max + geometry.x_min),
                0.5 * (geometry.y_max + geometry.y_min),
            ),
            material=material,
        )
    if isinstance(geometry, Polygon2D):
        vertices = [mp.Vector3(float(x), float(y)) for x, y in geometry.poly]
        return mp.Prism(
            vertices=vertices,
            height=mp.inf,
            axis=mp.Vector3(z=1),
            material=material,
        )
    msg = f"Unsupported MEEP cross-section geometry: {type(geometry).__name__}."
    raise TypeError(msg)


def meep_material(material: Material, env: Environment) -> tuple[float, Any]:
    """(n, mp.Medium) for a meow material at this environment's wavelength.

    Asks the material for its own index — `Material.__call__(env)` — instead of
    reaching into `material.n` / `material.params["wl"]` and re-implementing the
    lookup here, which is what this did until 2026-07-28:

        idx = np.argmin(np.abs(wls - wl))    # nearest sample, not interpolated
        n = ns[idx]

    That was a nearest-neighbour snap over the raw table, and meow's own
    `SampledMaterial.__call__` already interpolates properly, so this bypassed
    a correct implementation for a worse one. On meow's silicon table it put a
    **step discontinuity in the middle of the C-band**: only two samples exist
    between 1.45 and 1.65 µm (1.5320 and 1.6000), so every wavelength up to
    their midpoint got n = 3.4784 and everything past it got 3.4710, jumping
    0.0063 between 1.56 and 1.57 µm. That is the +0.00022 kink in the EME's
    Δn_eff recorded as directional_coupler_validation.md item 8. Away from the
    step it was biased too — +0.00196 at 1.55 µm.

    Going through `__call__` also stops assuming the material *has* a sampled
    table, so an analytic material (a Sellmeier/Lorentzian model, say) works
    here unchanged — `params["wl"]` would have raised.
    """
    import meep as mp  # ty: ignore[unresolved-import]

    n = complex(np.asarray(material(env)).reshape(-1)[0])
    return n.real, mp.Medium(epsilon=n.real**2)
