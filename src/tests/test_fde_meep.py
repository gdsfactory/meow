"""Test the MEEP FDE backend."""

import gdsfactory as gf
import numpy as np
import pytest

import meow as mw

pytest.importorskip("meep")

gf.gpdk.PDK.activate()


def _straight_cross_section(width: float = 0.5, t_soi: float = 0.22) -> mw.CrossSection:
    c = gf.components.straight(length=1.0, width=width)
    extrusion_rules = {
        (1, 0): [
            mw.GdsExtrusionRule(
                material=mw.silicon,
                h_min=0.0,
                h_max=t_soi,
                mesh_order=1,
            ),
        ],
    }
    structs = mw.extrude_gds(c, extrusion_rules)
    mesh = mw.Mesh2D(
        x=np.linspace(-1.5, 1.5, 61),
        y=np.linspace(-1.0, 1.0, 41),
    )
    cell = mw.Cell(structures=structs, mesh=mesh, z_min=0.0, z_max=1.0)
    env = mw.Environment(wl=1.55, T=25.0)
    return mw.CrossSection.from_cell(cell=cell, env=env)


def test_compute_modes_meep_returns_guided_fundamental_mode():
    cs = _straight_cross_section()

    modes = mw.compute_modes_meep(cs, num_modes=2)

    assert len(modes) >= 1
    neff = float(np.real(modes[0].neff))
    # Fundamental mode of a silicon strip waveguide near 1.55um must be
    # guided (above cladding index) and below silicon's bulk index.
    assert 1.4 < neff < 3.48
    assert modes[0].Ex.shape == (len(cs.mesh.x_), len(cs.mesh.y_))


def test_compute_modes_meep_rejects_non_square_mesh():
    c = gf.components.straight(length=1.0, width=0.5)
    extrusion_rules = {
        (1, 0): [
            mw.GdsExtrusionRule(
                material=mw.silicon, h_min=0.0, h_max=0.22, mesh_order=1
            ),
        ],
    }
    structs = mw.extrude_gds(c, extrusion_rules)
    mesh = mw.Mesh2D(
        x=np.linspace(-1.5, 1.5, 61),
        y=np.linspace(-1.0, 1.0, 21),  # different pixel spacing than x
    )
    cell = mw.Cell(structures=structs, mesh=mesh, z_min=0.0, z_max=1.0)
    env = mw.Environment(wl=1.55, T=25.0)
    cs = mw.CrossSection.from_cell(cell=cell, env=env)

    with pytest.raises(ValueError, match="square mesh"):
        mw.compute_modes_meep(cs, num_modes=1)
