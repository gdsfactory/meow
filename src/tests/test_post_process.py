import numpy as np
from mode_data import MODE_DATA

import meow as mw
from meow.fde.post_process import orthonormalize_modes
from meow.mode import Mode


def _mode_with_field(ex: np.ndarray, neff: complex) -> Mode:
    reference = Mode.model_validate(MODE_DATA)
    zeros = np.zeros_like(reference.Ex)
    return Mode(
        cs=reference.cs,
        neff=neff,
        Ex=ex,
        Ey=zeros,
        Ez=zeros,
        Hx=zeros,
        Hy=zeros,
        Hz=zeros,
    )


def _l2_inner_product(mode1: Mode, mode2: Mode) -> complex:
    return complex(np.vdot(mode1.Ex, mode2.Ex))


def _nonorthogonal_modes(neff1: complex, neff2: complex) -> list[Mode]:
    shape = Mode.model_validate(MODE_DATA).Ex.shape
    ex1 = np.zeros(shape, dtype=complex)
    ex2 = np.zeros(shape, dtype=complex)
    ex1.flat[0] = 1
    ex2.flat[:2] = 1
    return [_mode_with_field(ex1, neff1), _mode_with_field(ex2, neff2)]


def test_orthonormalize_modes_does_not_mix_non_degenerate_modes() -> None:
    modes = _nonorthogonal_modes(2.0, 1.5)

    result = orthonormalize_modes(modes, _l2_inner_product)

    assert len(result) == 2
    assert [mode.neff for mode in result] == [2.0, 1.5]
    assert np.isclose(_l2_inner_product(result[0], result[0]), 1)
    assert np.isclose(_l2_inner_product(result[1], result[1]), 1)
    assert not np.isclose(_l2_inner_product(result[0], result[1]), 0)


def test_orthonormalize_modes_orthogonalizes_degenerate_modes() -> None:
    neffs = [2.0, 2.0 + 5e-7]
    modes = _nonorthogonal_modes(*neffs)

    result = orthonormalize_modes(modes, _l2_inner_product)

    assert len(result) == 2
    assert [mode.neff for mode in result] == neffs
    overlaps = np.array([[_l2_inner_product(a, b) for b in result] for a in result])
    assert np.allclose(overlaps, np.eye(2))


def _box(x_min: float, x_max: float, y_min: float, y_max: float) -> mw.Box:
    return mw.Box(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, z_min=0, z_max=1)


def _compute_fundamental_mode(core_boxes: list[mw.Box]) -> Mode:
    structures = [
        mw.Structure(
            geometry=_box(-1, 1, -0.5, 1), material=mw.silicon_oxide, mesh_order=2
        ),
        *(
            mw.Structure(geometry=box, material=mw.silicon, mesh_order=1)
            for box in core_boxes
        ),
    ]
    cell = mw.Cell(
        structures=structures,
        z_min=0,
        z_max=1,
        mesh=mw.Mesh2D(x=np.linspace(-1, 1, 100), y=np.linspace(-0.5, 1, 100)),
    )
    cs = mw.CrossSection.from_cell(cell=cell, env=mw.Environment())
    return mw.compute_modes(cs, num_modes=1)[0]


def _dominant_e_field_at_energy_max(mode: Mode) -> tuple[str, complex]:
    energy = np.abs(mw.energy_density(mode))
    m, n = np.argwhere(energy == energy.max())[0]
    values = {c: complex(getattr(mode, c)[m, n]) for c in ("Ex", "Ey", "Ez")}
    name = max(values, key=lambda c: abs(values[c]))
    return name, values[name]


def _assert_real_positive(value: complex) -> None:
    assert value.real > 0
    assert np.isclose(value.imag, 0, atol=1e-6 * abs(value))


def test_zero_phase_makes_largest_field_positive_rib_waveguide() -> None:
    mode = _compute_fundamental_mode(
        [_box(-1, 1, 0, 0.15), _box(-0.25, 0.25, 0.15, 0.22)]  # slab + ridge
    )

    name, value = _dominant_e_field_at_energy_max(mode)

    assert name == "Ex"  # fundamental mode is polarized mostly along x
    _assert_real_positive(value)


def test_zero_phase_makes_largest_field_positive_tall_waveguide() -> None:
    # taller than wide: the fundamental mode is polarized mostly along y
    mode = _compute_fundamental_mode([_box(-0.15, 0.15, 0.0, 0.8)])

    name, value = _dominant_e_field_at_energy_max(mode)

    assert name == "Ey"
    _assert_real_positive(value)
