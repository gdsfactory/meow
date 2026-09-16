"""Test the MEEP FDE backend."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any, ClassVar

import gdsfactory as gf
import numpy as np
import pytest

import meow as mw
from meow.fde.meep import _meep_geometry

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
    pytest.importorskip("meep")
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


def test_compute_modes_meep_rejects_nonuniform_mesh():
    cs = _straight_cross_section()
    x = np.array(cs.mesh.x, copy=True)
    x[2] += 0.01
    cs = cs.model_copy(update={"mesh": cs.mesh.model_copy(update={"x": x})})

    with pytest.raises(ValueError, match="uniform square mesh"):
        mw.compute_modes_meep(cs, num_modes=1)


def test_compute_modes_meep_preserves_complex_fields_and_resets(
    monkeypatch: pytest.MonkeyPatch,
):
    class Vector3:
        def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
            self.values = (x, y, z)

        def __getitem__(self, index: int) -> float:
            return self.values[index]

    class Eigenmode:
        k = Vector3(z=1.0)

        @staticmethod
        def amplitude(*, point: Any, component: Any) -> complex:  # noqa: ARG004
            return 1.0 + 2.0j

    class Simulation:
        instance: ClassVar[Simulation | None] = None

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.was_reset = False
            Simulation.instance = self

        def init_sim(self) -> None:
            return None

        def get_eigenmode(self, **_kwargs: Any) -> Eigenmode:
            return Eigenmode()

        def reset_meep(self) -> None:
            self.was_reset = True

    def namespace(**kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(**kwargs)

    fake_meep = SimpleNamespace(
        Block=namespace,
        Medium=namespace,
        Prism=namespace,
        Simulation=Simulation,
        Vector3=Vector3,
        Volume=namespace,
        Ex="Ex",
        Ey="Ey",
        Ez="Ez",
        Hx="Hx",
        Hy="Hy",
        Hz="Hz",
        NO_DIRECTION=0,
        NO_PARITY=0,
        inf=np.inf,
        verbosity=lambda _level: None,
    )
    monkeypatch.setitem(sys.modules, "meep", fake_meep)
    processed: list[mw.Mode] = []

    def post_process(modes: list[mw.Mode]) -> list[mw.Mode]:
        processed.extend(modes)
        return modes

    modes = mw.compute_modes_meep(
        _straight_cross_section(), num_modes=1, post_process=post_process
    )

    assert modes == processed
    assert np.all(modes[0].Ex.imag == 2.0)
    assert Simulation.instance is not None
    assert Simulation.instance.was_reset


def test_meep_geometry_supports_polygons(monkeypatch: pytest.MonkeyPatch):
    class Vector3:
        def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
            self.values = (x, y, z)

    def namespace(**kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(**kwargs)

    fake_meep = SimpleNamespace(
        Medium=namespace,
        Prism=namespace,
        Vector3=Vector3,
        inf=np.inf,
    )
    monkeypatch.setitem(sys.modules, "meep", fake_meep)
    structure = mw.Structure(
        material=mw.silicon,
        geometry=mw.Polygon2D(poly=np.array([[0, 0], [1, 0], [0, 1]])),
    )

    geometry = _meep_geometry(structure, mw.Environment(wl=1.55, T=25.0), fake_meep)

    assert len(geometry.vertices) == 3
    assert geometry.height == np.inf


def test_compute_modes_tidy3d_honors_post_process():
    expected = []

    result = mw.compute_modes_tidy3d(
        _straight_cross_section(), num_modes=1, post_process=lambda _modes: expected
    )

    assert result is expected
