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
import meep as mp

from meow.cross_section import CrossSection
from meow.fde.post_process import post_process_modes
from meow.mode import Mode, Modes, inner_product, normalize

mp.verbosity(0) # Suppress MEEP output

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

    # Assume all structures are rectangles
    x_min = 0
    x_max = 0
    y_min = 0
    y_max = 0
    geometry_waveguide = []
    geometry_oxide = []
    for struct in cs.structures:
        n, material = meep_material(struct.material.n, struct.material.params["wl"], cs.env.wl)
        if x_min > struct.geometry.x_min:
            x_min = struct.geometry.x_min
        if x_max < struct.geometry.x_max:
            x_max = struct.geometry.x_max
        if y_min > struct.geometry.y_min:
            y_min = struct.geometry.y_min
        if y_max < struct.geometry.y_max:
            y_max = struct.geometry.y_max
        if n > 2: # Assume silicon if n > 2, otherwise assume oxide
            geometry_waveguide += [mp.Block(size=mp.Vector3(struct.geometry.x_max - struct.geometry.x_min, struct.geometry.y_max - struct.geometry.y_min, mp.inf), center=mp.Vector3(struct.geometry.x_max + struct.geometry.x_min, struct.geometry.y_max + struct.geometry.y_min, 0)/2, material=material)]
        else:
            geometry_oxide += [mp.Block(size=mp.Vector3(struct.geometry.x_max - struct.geometry.x_min, struct.geometry.y_max - struct.geometry.y_min, mp.inf), center=mp.Vector3(struct.geometry.x_max + struct.geometry.x_min, struct.geometry.y_max + struct.geometry.y_min, 0)/2, material=material)]

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
        resolution=1/dx,
    )
    geometry_lattice = mp.Volume(
        center = (
            mp.Vector3(x_center, y_center, 0)
        ),
        size = (
            mp.Vector3(
                x_span,
                y_span,
                0
            )
        )
    )
    sim.init_sim()

    modes = []
    for mode_num in range(1, num_modes+1):
        # Calculate the mode data for the given mode number
        mode_data = sim.get_eigenmode(
            frequency = 1/cs.env.wl ,
            direction = mp.NO_DIRECTION,
            where = geometry_lattice,
            band_num = mode_num,
            parity = mp.NO_PARITY,
            kpoint = mp.Vector3(z=1),
            # resolution = 1/(cs.mesh.x[1] - cs.mesh.x[0]),
            eigensolver_tol = 1e-12
        )
        # Sample on the CrossSection's own cell-centered grid (matches
        # cs.mesh.x_/y_ exactly, so downstream inner_product()/normalize()
        # shapes align).
        y = cs.mesh.y_
        x = cs.mesh.x_
        Ex = np.zeros([Nx,Ny]) # arrays to store the data
        Ey = np.zeros([Nx,Ny]) # arrays to store the data
        Ez = np.zeros([Nx,Ny]) # arrays to store the data
        Hx = np.zeros([Nx,Ny]) # arrays to store the data
        Hy = np.zeros([Nx,Ny]) # arrays to store the data
        Hz = np.zeros([Nx,Ny]) # arrays to store the data
        for i in range(Nx):
            for j in range(Ny):
                Ex[i,j] = np.real(mode_data.amplitude(point=mp.Vector3(x[i],y[j]),component=mp.Ex))
                Ey[i,j] = np.real(mode_data.amplitude(point=mp.Vector3(x[i],y[j]),component=mp.Ey))
                Ez[i,j] = np.real(mode_data.amplitude(point=mp.Vector3(x[i],y[j]),component=mp.Ez))
                Hx[i,j] = np.real(mode_data.amplitude(point=mp.Vector3(x[i],y[j]),component=mp.Hx))
                Hy[i,j] = np.real(mode_data.amplitude(point=mp.Vector3(x[i],y[j]),component=mp.Hy))
                Hz[i,j] = np.real(mode_data.amplitude(point=mp.Vector3(x[i],y[j]),component=mp.Hz))
        # Get the effective index of the mode
        neff = mode_data.k[2] * cs.env.wl
        # Normalize and save the mode data in the modes list
        mode = Mode(
            cs=cs,
            Ex=Ex,
            Ey=Ey,
            Ez=Ez,
            Hx=Hx,
            Hy=Hy,
            Hz=Hz,
            neff=neff,
        )
        mode = normalize(mode, inner_product)
        modes.append(mode)

    modes = sorted(modes, key=lambda m: float(np.real(m.neff)), reverse=True)
    sim.reset_meep()  # free this simulation's grid/structure/PML now, not at interpreter shutdown
    return modes

# Function to get material of rectangle
def meep_material(ns, wls, wl):
    idx = np.argmin(np.abs(wls - wl))
    n = ns[idx]
    return n, mp.Medium(epsilon=n**2)