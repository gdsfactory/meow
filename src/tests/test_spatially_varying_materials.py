"""Tests for spatially varying materials."""

import itertools
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import meow as mw

visualize = False
"""Set by `main(visualize=True)` so the equivalence tests plot their modes.
"""


def _box(
    x_min: float = -1.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 0.5,
    z_min: float = 0.0,
    z_max: float = 1.0,
) -> mw.Box:
    return mw.Box(
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, z_min=z_min, z_max=z_max
    )


# The index callables below take the full (x, y, z) signature that
# SpatiallyVaryingMaterial calls positionally, so most of them deliberately
# ignore some arguments. Defining them as named helpers keeps that signature
# requirement in one place instead of on every lambda.
def _const(value: float) -> Callable[..., float]:
    """A spatially varying material callable returning a constant index."""

    def n(x: object, y: object, z: object) -> float:  # noqa: ARG001
        return value

    return n


def _ramp(offset: float = 1.0, slope: float = 0.1) -> Callable[..., np.ndarray]:
    """A callable returning ``offset + slope * x`` (vectorized in x)."""

    def n(x: np.ndarray, y: object, z: object) -> np.ndarray:  # noqa: ARG001
        return offset + slope * x

    return n


def _ramp_x_y(offset: float = 1.0, slope: float = 0.1) -> Callable[..., np.ndarray]:
    """A callable returning ``offset + slope * x + slope * y``

    Vectorized in x and y."""

    def n(x: np.ndarray, y: np.ndarray, z: object) -> np.ndarray:  # noqa: ARG001
        return offset + slope * x + slope * y

    return n


def _plus_z(offset: float = 2.0) -> Callable[..., float]:
    """A callable returning ``offset + z``; used to test z propagation."""

    def n(x: object, y: object, z: float) -> float:  # noqa: ARG001
        return offset + z

    return n


def _cross_section(
    material: mw.MaterialBase,
    *,
    z_min: float = 0.0,
    z_max: float = 1.0,
    n: int = 51,
    subpixel_smoothing: bool = True,
) -> mw.CrossSection:
    """Make a CrossSection with just one box containing the given material.

    The box is guaranteed to overlap the center of the cell's z range.
    """
    mesh = mw.Mesh2D(x=np.linspace(-1.5, 1.5, n), y=np.linspace(-1.0, 2.0, n))
    # the box must span the cell's z-range, otherwise the projection at the
    # cell center finds no geometry at all
    struct = mw.Structure3D(
        material=material,
        geometry=_box(z_min=min(z_min, 0.0) - 1.0, z_max=max(z_max, 1.0) + 1.0),
        mesh_order=2,
    )
    cell = mw.Cell(structures=[struct], mesh=mesh, z_min=z_min, z_max=z_max)
    return mw.CrossSection.from_cell(
        cell=cell, env=mw.Environment(wl=1.55), subpixel_smoothing=subpixel_smoothing
    )


# --- construction / validation ---
# These tests just test that SpatiallyVaryingMaterial constructor
# validates inputs properly.


def test_both_isotropic_and_anisotropic_is_rejected() -> None:
    with pytest.raises(ValueError, match="not both"):
        mw.SpatiallyVaryingMaterial(
            name="both",
            n=_const(1.0),
            nx=_const(1.0),
            ny=_const(2.0),
            nz=_const(3.0),
        )


def test_no_callable_is_rejected() -> None:
    with pytest.raises(ValueError, match="must specify either"):
        mw.SpatiallyVaryingMaterial(name="none")


def test_partial_anisotropic_is_rejected() -> None:
    with pytest.raises(ValueError, match="all of 'nx', 'ny' and 'nz'"):
        mw.SpatiallyVaryingMaterial(name="partial", nx=_const(1.0))


def test_non_callable_is_rejected() -> None:
    with pytest.raises(Exception, match="callable"):
        mw.SpatiallyVaryingMaterial(name="notcall", n=2.0)


def test_call_raises_because_coordinates_are_required() -> None:
    """`__call__` needs coordinates; it must not return a meaningless scalar."""
    mat = mw.SpatiallyVaryingMaterial(name="iso", n=_const(2.0))
    with pytest.raises(NotImplementedError, match="n_at"):
        mat(mw.Environment())


# --- evaluation ---


def test_n_at_evaluates_user_function() -> None:
    mat = mw.SpatiallyVaryingMaterial(name="iso", n=_ramp())
    x = np.array([0.0, 1.0, 2.0])
    y = np.zeros(3)
    result = mat.n_at(mw.Environment(), x, y, 0.0)
    assert np.allclose(result, [1.0, 1.1, 1.2])


def test_n_at_error_no_broadcast() -> None:
    mat = mw.SpatiallyVaryingMaterial(name="iso", n=_ramp_x_y())
    x = np.zeros(3)
    y = np.zeros(2)
    with pytest.raises(ValueError):
        mat.n_at(mw.Environment(), x, y, 0.0)


# --- cross-section integration ---


def test_n_full_linear_ramp_is_exact() -> None:
    """n = 1 + 0.1x must be reproduced exactly where the material is present."""
    mat = mw.SpatiallyVaryingMaterial(name="ramp", n=_ramp())
    cs = _cross_section(mat)
    n_full = cs.n_full("x")
    mask = cs._m_full > 0
    expected = 1.0 + 0.1 * cs.mesh.X_full
    assert np.all(np.real(n_full[mask]) == expected[mask])
    # background is untouched
    assert np.all(np.real(n_full[~mask]) == 1.0)


def test_n_full_uses_cell_z() -> None:
    """A material that varies along z is sampled at the cell centre."""
    mat = mw.SpatiallyVaryingMaterial(name="zmat", n=_plus_z())
    # cell centre is z = 2.0
    cs = _cross_section(mat, z_min=1.0, z_max=3.0)
    assert cs.z == 2.0
    assert set(np.round(np.real(cs.n_full()), 6).ravel()) == {1.0, 4.0}

    cs2 = _cross_section(mat, z_min=5.0, z_max=7.0)
    assert cs2.z == 6.0
    assert set(np.round(np.real(cs2.n_full()), 6).ravel()) == {1.0, 8.0}


def test_scalar_returning_callable_broadcasts() -> None:
    """A callable that returns a scalar is allowed, even when handed arrays.

    This is the simplest possible user function and must work on every path:
    the result has to be broadcast over the region rather than kept as a 0-d
    array, which would fail on boolean indexing inside the smoothing code.
    """

    def scalar(x: np.ndarray, y: object, z: object) -> float:  # noqa: ARG001
        return 2.0

    mat = mw.SpatiallyVaryingMaterial(name="scalar", n=scalar)
    for subpixel_smoothing in (True, False):
        cs = _cross_section(mat, subpixel_smoothing=subpixel_smoothing)
        assert set(np.round(np.real(cs.n_full()), 6).ravel()) == {1.0, 2.0}

        # cs.nw may have subpixel smoothing, so we can only ensure that refractive
        # index values are within the expected range rather than exact.
        n = np.real(cs.nx)
        assert n.max() == 2.0
        assert n.min() == 1.0


def test_partial_scalar_broadcast_over_region() -> None:
    """A constant callable filling part of the mesh must broadcast only there."""
    mat = mw.SpatiallyVaryingMaterial(name="scalar2", n=_const(3.0))
    cs = _cross_section(mat)
    n_full = np.real(cs.n_full("x"))
    mask = cs._m_full > 0
    assert np.all(n_full[mask] == 3.0)
    assert np.all(n_full[~mask] == 1.0)


def test_direct_construction_has_none_z_and_material_can_raise() -> None:
    """Without a cell there is no z; a z-dependent material must fail loudly."""
    mat = mw.SpatiallyVaryingMaterial(name="zmat", n=_plus_z())
    mesh = mw.Mesh2D(x=np.linspace(-1, 1, 11), y=np.linspace(-1, 1, 11))
    struct = mw.Structure2D(
        material=mat,
        geometry=mw.Rectangle(x_min=-0.5, x_max=0.5, y_min=-0.5, y_max=0.5),
    )
    cs = mw.CrossSection(structures=[struct], mesh=mesh, env=mw.Environment())
    assert cs.z is None
    with pytest.raises(TypeError):
        cs.n_full("x")


def test_direct_construction_z_can_be_given() -> None:
    mat = mw.SpatiallyVaryingMaterial(name="zmat", n=_plus_z())
    mesh = mw.Mesh2D(x=np.linspace(-1, 1, 11), y=np.linspace(-1, 1, 11))
    struct = mw.Structure2D(
        material=mat,
        geometry=mw.Rectangle(x_min=-0.5, x_max=0.5, y_min=-0.5, y_max=0.5),
    )
    cs = mw.CrossSection(structures=[struct], mesh=mesh, env=mw.Environment(), z=4.0)
    assert set(np.round(np.real(cs.n_full()), 6).ravel()) == {1.0, 6.0}


def test_anisotropic_varying_components_differ() -> None:
    mat = mw.SpatiallyVaryingMaterial(
        name="aniso",
        nx=_const(1.0),
        ny=_const(2.0),
        nz=_const(3.0),
    )
    cs = _cross_section(mat)
    # interior (away from the smoothed interface) holds the exact values
    assert np.isclose(np.real(cs.n_full("x")).max(), 1.0)
    assert np.isclose(np.real(cs.n_full("y")).max(), 2.0)
    assert np.isclose(np.real(cs.n_full("z")).max(), 3.0)


@pytest.mark.parametrize("subpixel_smoothing", [True, False])
def test_spatially_varying_boundary_is_smoothed(
    subpixel_smoothing: bool,  # noqa: FBT001
) -> None:
    """At an interface between a varying and a fixed material the varying
    material must contribute its *locally sampled* value.

    With subpixel smoothing on, the two materials are blended, so a value
    strictly between n=2 and n=4 must appear. Without smoothing the winner
    takes all, so only the two bulk values appear.

    Also covers the winner-takes-all path, which previously had no coverage.

    The two boxes are adjacent rather than nested: the LOWEST mesh order wins
    the *whole* overlap region (`_create_full_material_array` applies last-wins
    over a descending sort), so nesting one inside the other would erase it
    unless the inner one had a lower mesh_order.
    """
    ramp = mw.SpatiallyVaryingMaterial(name="spatially_varying_const", n=_const(2.0))
    fixed = mw.IndexMaterial(n=4.0, name="fixed")
    mesh = mw.Mesh2D(x=np.linspace(-1.0, 1.0, 41), y=np.linspace(-1.0, 1.0, 41))
    # The shared edge must fall *between* mesh points, otherwise no dual cell
    # straddles it and there is correctly nothing to smooth.
    structs = [
        mw.Structure3D(
            material=ramp, geometry=_box(-0.6, 0.007, -1.0, 1.0), mesh_order=2
        ),
        mw.Structure3D(
            material=fixed, geometry=_box(0.007, 0.6, -1.0, 1.0), mesh_order=1
        ),
    ]
    cell = mw.Cell(structures=structs, mesh=mesh, z_min=0, z_max=1)
    cs = mw.CrossSection.from_cell(
        cell=cell, env=mw.Environment(wl=1.55), subpixel_smoothing=subpixel_smoothing
    )
    # both materials are actually present (0 is the vacuum background)
    assert {1, 2} <= set(np.unique(cs._m_full))
    n = np.real(cs.nx)
    between = ((n > 2.0 + 1e-9) & (n < 4.0 - 1e-9)).sum()
    if subpixel_smoothing:
        assert between > 0, "expected blended values between n=2 and n=4"
    else:
        assert between == 0, "winner-takes-all must not blend"
    # the two bulk values are present either way
    assert np.isclose(n[n >= 2.0 - 1e-9].min(), 2.0)
    assert np.isclose(n.max(), 4.0)


def test_varying_material_used_as_dict_key() -> None:
    """Materials are keys in the material->index maps; a varying material must
    remain usable as a dict key so meshing still works."""
    mat = mw.SpatiallyVaryingMaterial(name="ramp", n=_ramp())
    assert isinstance(hash(mat), int)
    assert {mat: 1}[mat] == 1


# --- equivalence with discrete structures ---
#
# Three side-by-side boxes (indices 2, 4, 3) versus one box whose index
# callable reproduces the same layout.


def _step_mesh() -> mw.Mesh2D:
    return mw.Mesh2D(x=np.linspace(-2.0, 2.0, 101), y=np.linspace(-1.5, 1.5, 101))


def _step_index(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:  # noqa: ARG001
    """Return 2, 4 or 3 depending on x, matching the three boxes below."""
    return np.where(x < -0.5, 2.0, np.where(x < 0.5, 4.0, 3.0))


def _three_structures(mesh: mw.Mesh2D, *, smooth: bool) -> mw.CrossSection:
    """Three side-by-side boxes with constant indices 2, 4, 3."""
    structures = [
        mw.Structure3D(
            material=mw.IndexMaterial(n=2.0, name="n2"),
            geometry=mw.Box(
                x_min=-1.5, x_max=-0.5, y_min=-1.0, y_max=1.0, z_min=-1.0, z_max=1.0
            ),
            mesh_order=2,
        ),
        mw.Structure3D(
            material=mw.IndexMaterial(n=4.0, name="n3"),
            geometry=mw.Box(
                x_min=-0.5, x_max=0.5, y_min=-1.0, y_max=1.0, z_min=-1.0, z_max=1.0
            ),
            mesh_order=2,
        ),
        mw.Structure3D(
            material=mw.IndexMaterial(n=3.0, name="n4"),
            geometry=mw.Box(
                x_min=0.5, x_max=1.5, y_min=-1.0, y_max=1.0, z_min=-1.0, z_max=1.0
            ),
            mesh_order=2,
        ),
    ]
    cell = mw.Cell(structures=structures, mesh=mesh, z_min=-1.0, z_max=1.0)
    return mw.CrossSection.from_cell(
        cell=cell, env=mw.Environment(wl=1.55), subpixel_smoothing=smooth
    )


def _varying(mesh: mw.Mesh2D, *, smooth: bool) -> mw.CrossSection:
    """One box whose index callable reproduces the three-column layout above."""
    material = mw.SpatiallyVaryingMaterial(name="stepped", n=_step_index)
    structure = mw.Structure3D(
        material=material,
        geometry=mw.Box(
            x_min=-1.5, x_max=1.5, y_min=-1.0, y_max=1.0, z_min=-1.0, z_max=1.0
        ),
        mesh_order=2,
    )
    cell = mw.Cell(structures=[structure], mesh=mesh, z_min=-1.0, z_max=1.0)
    return mw.CrossSection.from_cell(
        cell=cell, env=mw.Environment(wl=1.55), subpixel_smoothing=smooth
    )


def test_discrete_and_varying_agree_with_smoothing() -> None:
    """With subpixel smoothing on, the two descriptions nearly agree.

    They cannot agree exactly: the internal 2|4 and 4|3 junctions are material
    boundaries in the discrete case (so smoothed) but interior to a single
    material in the varying case (so not smoothed). The raw index grids are
    identical; only the smoothed grids differ, and only at those junctions.
    """
    mesh = _step_mesh()
    cs_three_boxes = _three_structures(mesh, smooth=True)
    cs_varying = _varying(mesh, smooth=True)

    # the underlying material definition is exactly the same
    assert np.array_equal(cs_three_boxes.n_full("x"), cs_varying.n_full("x"))

    [mode_three_boxes] = mw.compute_modes(cs_three_boxes, num_modes=1)
    [mode_varying] = mw.compute_modes(cs_varying, num_modes=1)

    if visualize:
        import matplotlib.pyplot as plt

        _fig, ax = _comparison_figure()
        mw.visualize(cs_three_boxes, ax=ax[0][0], show=False)
        ax[0][0].set_title("n, boxes")
        mw.visualize(cs_varying, ax=ax[0][1], show=False)
        ax[0][1].set_title("n, spatially varying material")
        mw.visualize(mode_three_boxes, fields=["Ex"], ax=ax[1][0], show=False)
        mw.visualize(mode_varying, fields=["Ex"], ax=ax[1][1], show=False)
        mw.visualize(mode_three_boxes, fields=["Ey"], ax=ax[2][0], show=False)
        mw.visualize(mode_varying, fields=["Ey"], ax=ax[2][1], show=False)
        plt.show()

    overlap_sq = np.abs(mw.inner_product(mode_three_boxes, mode_varying)) ** 2
    assert overlap_sq > 0.999
    assert abs(mode_three_boxes.neff - mode_varying.neff) < 1e-3


def test_discrete_and_varying_bit_for_bit_without_smoothing() -> None:
    """With subpixel smoothing off, the two are bit-for-bit identical.

    The winner-takes-all rule assigns each pixel a material's own permittivity,
    so the varying material -- which returns exactly the same indices as the
    three boxes -- produces an identical grid, and hence identical modes.

    Note the smoothing is deliberately OFF here. With it ON the two cannot be
    bit-identical, for the reason given in the previous test.
    """
    mesh = _step_mesh()
    cs_three_boxes = _three_structures(mesh, smooth=False)
    cs_varying = _varying(mesh, smooth=False)

    assert np.array_equal(np.real(cs_three_boxes.nx), np.real(cs_varying.nx))
    assert np.array_equal(np.real(cs_three_boxes.ny), np.real(cs_varying.ny))
    assert np.array_equal(np.real(cs_three_boxes.nz), np.real(cs_varying.nz))

    [mode_three_boxes] = mw.compute_modes(cs_three_boxes, num_modes=1)
    [mode_varying] = mw.compute_modes(cs_varying, num_modes=1)

    if visualize:
        import matplotlib.pyplot as plt

        _fig, ax = _comparison_figure()
        mw.visualize(cs_three_boxes, ax=ax[0][0], show=False)
        ax[0][0].set_title("n, boxes")
        mw.visualize(cs_varying, ax=ax[0][1], show=False)
        ax[0][1].set_title("n, spatially varying material")
        mw.visualize(mode_three_boxes, fields=["Ex"], ax=ax[1][0], show=False)
        mw.visualize(mode_varying, fields=["Ex"], ax=ax[1][1], show=False)
        mw.visualize(mode_three_boxes, fields=["Ey"], ax=ax[2][0], show=False)
        mw.visualize(mode_varying, fields=["Ey"], ax=ax[2][1], show=False)
        plt.show()

    assert mode_three_boxes.neff == mode_varying.neff
    assert np.array_equal(mode_three_boxes.Ex, mode_varying.Ex)
    assert np.array_equal(mode_three_boxes.Ey, mode_varying.Ey)
    assert np.array_equal(mode_three_boxes.Ez, mode_varying.Ez)


# --- anisotropic equivalence with discrete structures ---
#
# Same idea as above, but biaxial. The X-profile (2, 4, 3) varies along x and
# goes into nx; the Y-profile (1.99, 3.99, 2.99) varies along y and goes into
# ny. Very close to X-profile but slightly different so we can distinguish
# x and y polarized modes by effective index.

_X_PROFILE = (2.0, 4.0, 3.0)
_Y_PROFILE = (1.99, 3.99, 2.99)
_NZ = 9.0

# region edges, shared by both descriptions.
#
# Two details matter for the exact comparison:
#
# 1. The callables use half-open intervals ``[lo, hi)``. A Box owns its
#    ``x_min`` but shares its ``x_max`` with the next box, so the boundary
#    point belongs to the *next* region. Using ``<=`` would put each upper
#    boundary row in the lower region and break the comparison.
#
# 2. The constants must be the exact floats the mesh uses, not nearby literals.
#    ``np.linspace(-1.5, 1.5, 201)`` gives 0.50999999999999979 near +0.51, which
#    is strictly less than the literal 0.51; comparing against the literal
#    would misassign that boundary row by one float ulp.
_Y_EDGE_LO = -0.51
_Y_EDGE_HI = 0.5099999999999998
_X_EDGE_LO = -0.52
_X_EDGE_HI = 0.52

_X_EDGES = (-1.48, _X_EDGE_LO, _X_EDGE_HI, 1.48)
_Y_EDGES = (-1.02, _Y_EDGE_LO, _Y_EDGE_HI, 1.02)


def _aniso_step_index_x(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:  # noqa: ARG001
    """Return the X-profile values depending on x (half-open regions)."""
    return np.where(
        x < _X_EDGE_HI,
        np.where(x < _X_EDGE_LO, _X_PROFILE[0], _X_PROFILE[1]),
        _X_PROFILE[2],
    )


def _aniso_step_index_y(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:  # noqa: ARG001
    """Return the Y-profile values depending on y (half-open regions)."""
    return np.where(
        y < _Y_EDGE_HI,
        np.where(y < _Y_EDGE_LO, _Y_PROFILE[0], _Y_PROFILE[1]),
        _Y_PROFILE[2],
    )


def _aniso_boxes(mesh: mw.Mesh2D, *, smooth: bool) -> mw.CrossSection:
    """A 3x3 grid of boxes with constant anisotropic indices."""
    structures = []
    for i, (x_lo, x_hi) in enumerate(itertools.pairwise(_X_EDGES)):
        for j, (y_lo, y_hi) in enumerate(itertools.pairwise(_Y_EDGES)):
            material = mw.AnisotropicIndexMaterial(
                name=f"aniso_{i}{j}",
                nx=_X_PROFILE[i],
                ny=_Y_PROFILE[j],
                nz=_NZ,
            )
            structures.append(
                mw.Structure3D(
                    material=material,
                    geometry=mw.Box(
                        x_min=x_lo,
                        x_max=x_hi,
                        y_min=y_lo,
                        y_max=y_hi,
                        z_min=-1.0,
                        z_max=1.0,
                    ),
                    mesh_order=2,
                )
            )
    cell = mw.Cell(structures=structures, mesh=mesh, z_min=-1.0, z_max=1.0)
    return mw.CrossSection.from_cell(
        cell=cell, env=mw.Environment(wl=1.55), subpixel_smoothing=smooth
    )


def _aniso_varying(mesh: mw.Mesh2D, *, smooth: bool) -> mw.CrossSection:
    """One box whose callables reproduce the 3x3 anisotropic layout."""
    material = mw.SpatiallyVaryingMaterial(
        name="aniso_stepped",
        nx=_aniso_step_index_x,
        ny=_aniso_step_index_y,
        nz=_const(_NZ),
    )
    structure = mw.Structure3D(
        material=material,
        geometry=mw.Box(
            x_min=_X_EDGES[0],
            x_max=_X_EDGES[-1],
            y_min=_Y_EDGES[0],
            y_max=_Y_EDGES[-1],
            z_min=-1.0,
            z_max=1.0,
        ),
        mesh_order=2,
    )
    cell = mw.Cell(structures=[structure], mesh=mesh, z_min=-1.0, z_max=1.0)
    return mw.CrossSection.from_cell(
        cell=cell, env=mw.Environment(wl=1.55), subpixel_smoothing=smooth
    )


def test_anisotropic_discrete_and_varying_agree_with_smoothing() -> None:
    """With subpixel smoothing on, the biaxial descriptions nearly agree.

    As in the isotropic case they cannot agree exactly, because the internal
    region boundaries are material boundaries in the discrete description (so
    smoothed) but interior to a single material in the varying one.
    """
    mesh = _step_mesh()
    cs_boxes = _aniso_boxes(mesh, smooth=True)
    cs_varying = _aniso_varying(mesh, smooth=True)

    # the underlying material definition is exactly the same, per component
    for component in ("x", "y", "z"):
        assert np.array_equal(cs_boxes.n_full(component), cs_varying.n_full(component))

    modes_boxes = mw.compute_modes(cs_boxes, num_modes=2)
    modes_varying = mw.compute_modes(cs_varying, num_modes=2)

    if visualize:
        import matplotlib.pyplot as plt

        mode_polarizations = ["x", "y"]

        for mode_index, polarization in enumerate(mode_polarizations):
            _fig, ax = _comparison_figure()
            mw.visualize(
                cs_boxes, ax=ax[0][0], index_component=polarization, show=False
            )
            ax[0][0].set_title(f"n_{polarization} boxes")
            mw.visualize(
                cs_varying, ax=ax[0][1], index_component=polarization, show=False
            )
            ax[0][1].set_title(f"n_{polarization} spatially varying material")
            mw.visualize(
                modes_boxes[mode_index], fields=["Ex"], ax=ax[1][0], show=False
            )
            mw.visualize(
                modes_varying[mode_index], fields=["Ex"], ax=ax[1][1], show=False
            )
            mw.visualize(
                modes_boxes[mode_index], fields=["Ey"], ax=ax[2][0], show=False
            )
            mw.visualize(
                modes_varying[mode_index], fields=["Ey"], ax=ax[2][1], show=False
            )
            plt.show()

    for mode_index in range(2):
        overlap_sq = (
            np.abs(mw.inner_product(modes_boxes[mode_index], modes_varying[mode_index]))
            ** 2
        )
        assert overlap_sq > 0.999
        assert abs(modes_boxes[mode_index].neff - modes_varying[mode_index].neff) < 1e-3


def test_anisotropic_discrete_and_varying_bit_for_bit_without_smoothing() -> None:
    """With subpixel smoothing off, the biaxial descriptions are bit-for-bit
    identical."""
    mesh = _step_mesh()
    cs_boxes = _aniso_boxes(mesh, smooth=False)
    cs_varying = _aniso_varying(mesh, smooth=False)

    for component in ("x", "y", "z"):
        assert np.array_equal(
            np.real(cs_boxes.n_full(component)),
            np.real(cs_varying.n_full(component)),
        )

    assert np.array_equal(np.real(cs_boxes.nx), np.real(cs_varying.nx))
    assert np.array_equal(np.real(cs_boxes.ny), np.real(cs_varying.ny))
    assert np.array_equal(np.real(cs_boxes.nz), np.real(cs_varying.nz))

    modes_boxes = mw.compute_modes(cs_boxes, num_modes=2)
    modes_varying = mw.compute_modes(cs_varying, num_modes=2)

    if visualize:
        import matplotlib.pyplot as plt

        mode_polarizations = ["x", "y"]

        for mode_index, polarization in enumerate(mode_polarizations):
            _fig, ax = _comparison_figure()
            mw.visualize(
                cs_boxes, ax=ax[0][0], index_component=polarization, show=False
            )
            ax[0][0].set_title(f"n_{polarization} boxes")
            mw.visualize(
                cs_varying, ax=ax[0][1], index_component=polarization, show=False
            )
            ax[0][1].set_title(f"n_{polarization} spatially varying material")
            mw.visualize(
                modes_boxes[mode_index], fields=["Ex"], ax=ax[1][0], show=False
            )
            mw.visualize(
                modes_varying[mode_index], fields=["Ex"], ax=ax[1][1], show=False
            )
            mw.visualize(
                modes_boxes[mode_index], fields=["Ey"], ax=ax[2][0], show=False
            )
            mw.visualize(
                modes_varying[mode_index], fields=["Ey"], ax=ax[2][1], show=False
            )
            plt.show()

    for mode_index in range(2):
        assert modes_boxes[mode_index].neff == modes_varying[mode_index].neff
        assert np.array_equal(modes_boxes[mode_index].Ex, modes_varying[mode_index].Ex)
        assert np.array_equal(modes_boxes[mode_index].Ey, modes_varying[mode_index].Ey)
        assert np.array_equal(modes_boxes[mode_index].Ez, modes_varying[mode_index].Ez)


def test_anisotropic_varying_profiles_are_independent() -> None:
    """nx varies with x only, ny with y only, nz is constant.

    Guards against the two profiles being accidentally transposed or coupled.
    """
    mesh = _step_mesh()
    cs = _aniso_varying(mesh, smooth=False)

    nx = np.real(cs.n_full("x"))
    ny = np.real(cs.n_full("y"))
    nz = np.real(cs.n_full("z"))
    X = cs.mesh.X_full
    Y = cs.mesh.Y_full
    mask = cs._m_full > 0

    expected_x = np.where(
        X < _X_EDGE_HI,
        np.where(X < _X_EDGE_LO, _X_PROFILE[0], _X_PROFILE[1]),
        _X_PROFILE[2],
    )
    expected_y = np.where(
        Y < _Y_EDGE_HI,
        np.where(Y < _Y_EDGE_LO, _Y_PROFILE[0], _Y_PROFILE[1]),
        _Y_PROFILE[2],
    )

    assert np.allclose(nx[mask], expected_x[mask])
    assert np.allclose(ny[mask], expected_y[mask])
    assert np.allclose(nz[mask], _NZ)
    # the two profiles are genuinely different sets of values
    assert set(np.unique(nx[mask])) == set(_X_PROFILE)
    assert set(np.unique(ny[mask])) == set(_Y_PROFILE)


# --- serialization fails loudly ---


def test_serialization_raises() -> None:
    mat = mw.SpatiallyVaryingMaterial(name="ramp", n=_const(1.0))
    with pytest.raises(Exception, match="cannot be serialized"):
        mat.model_dump()


def test_pickling_raises() -> None:
    import pickle

    mat = mw.SpatiallyVaryingMaterial(name="ramp", n=_const(1.0))
    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(mat)


# --- manual inspection ---


def _comparison_figure(
    figsize: tuple[float, float] | None = None,
) -> tuple[Any, Any]:
    """A 3x2 grid for comparing two cross-sections and their two mode fields.

    ``constrained_layout`` is the fix for the overlapping labels: without it the
    subplots get no margins, so the mode titles collide with the row above and
    the colorbars are not allowed to take space from their own axes.

    ``figsize`` is left to matplotlib's default unless given, so a user's
    ``rcParams`` sizing still applies.
    """
    import matplotlib.pyplot as plt

    kwargs = {} if figsize is None else {"figsize": figsize}
    return plt.subplots(3, 2, constrained_layout=True, **kwargs)  # ty: ignore[no-matching-overload]


def main(visualize_modes: bool = False) -> None:  # noqa: FBT001, FBT002
    """Run the discrete-vs-varying equivalence checks for manual inspection.

    Args:
        visualize_modes: if True, set the module-level `visualize` flag so each
            test plots the modes it computes.
    """
    global visualize  # noqa: PLW0603
    visualize = visualize_modes

    tests = [
        test_discrete_and_varying_agree_with_smoothing,
        test_discrete_and_varying_bit_for_bit_without_smoothing,
        test_anisotropic_discrete_and_varying_agree_with_smoothing,
        test_anisotropic_discrete_and_varying_bit_for_bit_without_smoothing,
    ]
    failed = []
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failed.append(test.__name__)
            print(f"FAILED {test.__name__}: {exc}")  # noqa: T201
        else:
            print(f"passed {test.__name__}")  # noqa: T201

    if failed:
        msg = f"{len(failed)} of {len(tests)} tests failed: {failed}"
        raise SystemExit(msg)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="show the modes computed by each test",
    )
    args = parser.parse_args()
    main(visualize_modes=args.visualize)
