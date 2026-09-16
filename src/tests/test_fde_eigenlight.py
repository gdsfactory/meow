"""Test the EigenLight FDE backend."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import gdsfactory as gf
import numpy as np
import pytest

import meow as mw

gf.gpdk.PDK.activate()


def _straight_cross_section() -> mw.CrossSection:
    component = gf.components.straight(length=1.0, width=0.5)
    structures = mw.extrude_gds(
        component,
        {
            (1, 0): [
                mw.GdsExtrusionRule(
                    material=mw.silicon,
                    h_min=0.0,
                    h_max=0.22,
                    mesh_order=1,
                )
            ]
        },
    )
    mesh = mw.Mesh2D(
        x=np.linspace(-1.5, 1.5, 31),
        y=np.linspace(-1.0, 1.0, 21),
    )
    cell = mw.Cell(structures=structures, mesh=mesh, z_min=0.0, z_max=1.0)
    return mw.CrossSection.from_cell(
        cell=cell,
        env=mw.Environment(wl=1.55, T=25.0),
    )


def test_compute_modes_eigenlight_adapts_result(monkeypatch: pytest.MonkeyPatch):
    cs = _straight_cross_section()
    nx, ny = cs.nx.shape
    fields = np.full((2, nx, ny), 1.0 + 2.0j)
    calls: dict[str, Any] = {}

    def compute_modes(**kwargs: Any) -> SimpleNamespace:
        calls.update(kwargs)
        return SimpleNamespace(
            neff=np.array([2.4, 1.8]),
            keff=np.array([0.0, 0.01]),
            ex=fields,
            ey=fields,
            ez=fields,
            hx=fields,
            hy=fields,
            hz=fields,
        )

    monkeypatch.setitem(
        sys.modules,
        "eigenlight",
        SimpleNamespace(compute_modes=compute_modes),
    )
    processed: list[mw.Mode] = []

    def post_process(modes: list[mw.Mode]) -> list[mw.Mode]:
        processed.extend(modes)
        return modes

    modes = mw.compute_modes_eigenlight(
        cs,
        num_modes=2,
        target_neff=2.5,
        post_process=post_process,
    )

    assert modes == processed
    assert [mode.neff for mode in modes] == [2.4 + 0.0j, 1.8 + 0.01j]
    assert np.all(modes[0].Ex == 1.0 + 2.0j)
    assert calls["target_neff"] == 2.5
    assert calls["wavelength"] == cs.env.wl
    assert calls["eps_xx"].flags.c_contiguous


def test_compute_modes_eigenlight_integration():
    pytest.importorskip("eigenlight")

    modes = mw.compute_modes_eigenlight(
        _straight_cross_section(),
        num_modes=2,
        target_neff=2.5,
    )

    assert len(modes) == 2
    assert modes[0].Ex.shape == (30, 20)
    assert np.all(np.isfinite([mode.neff for mode in modes]))
