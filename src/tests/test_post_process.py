import numpy as np
from mode_data import MODE_DATA

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
