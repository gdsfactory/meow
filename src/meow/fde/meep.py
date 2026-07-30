"""FDE MEEP backend."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from pydantic import PositiveFloat, PositiveInt

from meow.cross_section import CrossSection
from meow.environment import Environment
from meow.fde.post_process import post_process_modes
from meow.materials import Material
from meow.mode import Mode, Modes, inner_product, normalize


def compute_modes_meep(
    cs: CrossSection,
    num_modes: PositiveInt = 10,
    target_neff: PositiveFloat | None = None,
    precision: Literal["single", "double"] = "double",
    post_process: Callable = post_process_modes,  # noqa: ARG001
) -> Modes:
    """Compute ``Modes`` for a given ``CrossSection``.

    Args:
        cs: the cross-section to solve modes for.
        num_modes: number of modes to compute.
        target_neff: effective index near which to search for modes.
        precision: floating-point precision, ``"single"`` or ``"double"``.
        post_process: accepted for parity with the other backends but not
            called (see the note at the end of this function).

    Returns:
        The computed collection of modes, individually normalized but not
        run through ``post_process``.
    """
    import meep as mp

    mp.verbosity(0)  # Suppress MEEP output

    if num_modes < 1:
        msg = "You need to request at least 1 mode."
        raise ValueError(msg)
    if target_neff is not None:
        msg = "compute_modes_meep does not yet support target_neff."
        raise NotImplementedError(msg)
    if precision != "double":
        msg = "compute_modes_meep does not yet support precision != 'double'."
        raise NotImplementedError(msg)

    # Assume all structures are rectangles
    geometry_waveguide = []
    geometry_oxide = []
    for struct in cs.structures:
        n, material = meep_material(struct.material, cs.env)
        if n > 2:  # Assume silicon if n > 2, otherwise assume oxide
            geometry_waveguide += [
                mp.Block(
                    size=mp.Vector3(
                        struct.geometry.x_max - struct.geometry.x_min,
                        struct.geometry.y_max - struct.geometry.y_min,
                        mp.inf,
                    ),
                    center=mp.Vector3(
                        struct.geometry.x_max + struct.geometry.x_min,
                        struct.geometry.y_max + struct.geometry.y_min,
                        0,
                    )
                    / 2,
                    material=material,
                )
            ]
        else:
            geometry_oxide += [
                mp.Block(
                    size=mp.Vector3(
                        struct.geometry.x_max - struct.geometry.x_min,
                        struct.geometry.y_max - struct.geometry.y_min,
                        mp.inf,
                    ),
                    center=mp.Vector3(
                        struct.geometry.x_max + struct.geometry.x_min,
                        struct.geometry.y_max + struct.geometry.y_min,
                        0,
                    )
                    / 2,
                    material=material,
                )
            ]

    # The mode-solve grid must match the CrossSection's own declared mesh
    # (cs.mesh.x / cs.mesh.y) rather than the raw bounding box of the
    # extruded structures: cladding layers routinely extend past the
    # intended mesh window (e.g. a larger vertical extrusion span than the
    # mesh's own y-extent), and deriving Nx/Ny from that mismatched bbox
    # produces a field grid whose shape doesn't match cs.mesh — breaking
    # inner_product()/normalize() downstream. mp.Block geometries are still
    # built from each structure's own true extent (a block wider than the
    # simulation cell is simply clipped by meep, which is correct).
    dx = cs.mesh.x[1] - cs.mesh.x[0]
    dy = cs.mesh.y[1] - cs.mesh.y[0]
    if not np.isclose(dx, dy):
        msg = (
            f"compute_modes_meep requires a square mesh (equal x/y pixel "
            f"spacing) since meep's resolution is isotropic; got dx={dx}, dy={dy}."
        )
        raise ValueError(msg)
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
        geometry=geometry_oxide + geometry_waveguide,
        eps_averaging=True,
        resolution=1 / dx,
    )
    geometry_lattice = mp.Volume(
        center=(mp.Vector3(x_center, y_center, 0)), size=(mp.Vector3(x_span, y_span, 0))
    )
    sim.init_sim()

    modes = []
    for mode_num in range(1, num_modes + 1):
        # Calculate the mode data for the given mode number
        mode_data = sim.get_eigenmode(
            frequency=1 / cs.env.wl,
            direction=mp.NO_DIRECTION,
            where=geometry_lattice,
            band_num=mode_num,
            parity=mp.NO_PARITY,
            kpoint=mp.Vector3(z=1),
            # resolution = 1/(cs.mesh.x[1] - cs.mesh.x[0]),
            eigensolver_tol=1e-12,
        )
        # Sample on the CrossSection's own cell-centered grid (matches
        # cs.mesh.x_/y_ exactly, so downstream inner_product()/normalize()
        # shapes align).
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
        fields = {name: np.zeros([Nx, Ny]) for name in components}
        for i in range(Nx):
            for j in range(Ny):
                point = mp.Vector3(x[i], y[j])
                for name, component in components.items():
                    fields[name][i, j] = np.real(
                        mode_data.amplitude(point=point, component=component)
                    )
        # Get the effective index of the mode
        neff = mode_data.k[2] * cs.env.wl
        # Normalize and save the mode data in the modes list
        mode = Mode(cs=cs, neff=neff, **fields)
        mode = normalize(mode, inner_product)
        modes.append(mode)

    modes = sorted(modes, key=lambda m: float(np.real(m.neff)), reverse=True)
    # free this simulation's grid/structure/PML now, not at interpreter shutdown
    sim.reset_meep()
    # Intentionally skip `post_process` here, matching compute_modes_tidy3d:
    # orthonormalize_modes()'s Gram-Schmidt step changes effective indices,
    # which breaks the EME solver. See the matching note in tidy3d.py.
    return modes


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
    import meep as mp

    n = complex(np.asarray(material(env)).reshape(-1)[0])
    return n.real, mp.Medium(epsilon=n.real**2)
