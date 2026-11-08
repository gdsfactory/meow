import numpy as np
import pytest

import meow as mw
from meow.materials import ComponentType

components: list[ComponentType] = ["x", "y", "z"]


class FakeMedium:
    """Used to model a medim with off-diagonal permittivity tensor elements.
    Since tidy3d does not yet support this, this is a "fake" class that has
    an eps_comp that can behave as such, letting you give it off-diagonal
    entries.
    """

    def __init__(self, permittivity: np.ndarray, frequency: float) -> None:
        self.permittivity = permittivity
        self.frequency = frequency

    def eps_comp(self, i: int, j: int, freq: float) -> complex:
        assert freq == self.frequency
        return self.permittivity[i, j]


def test_ensure_no_off_diagonal_accepts_diagonal_tensor():
    material = mw.TidyMaterial(name="LiNbO3", variant="Zelmon1997", optical_axis="z")
    frequency = 2e14
    medium = FakeMedium(np.diag([4.0, 5.0, 6.0]), frequency)

    material._ensure_no_off_diagonal(medium, frequency)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(("i", "j"), [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)])
@pytest.mark.parametrize("value", [0.1, -0.1, 0.1j])
def test_ensure_no_off_diagonal_rejects_off_diagonal_tensor(
    i: int, j: int, value: float
):
    material = mw.TidyMaterial(name="LiNbO3", variant="Zelmon1997", optical_axis="z")
    frequency = 2e14
    permittivity = np.diag([4.0, 5.0, 6.0]).astype(complex)
    permittivity[i, j] = value
    medium = FakeMedium(permittivity, frequency)

    with pytest.raises(ValueError, match="has off-diagonal permittivity components"):
        material._ensure_no_off_diagonal(medium, frequency)  # ty: ignore[invalid-argument-type]


def test_sampled_uniaxial_material_lithium_tantalate():
    from meow.materials import (
        lithium_tantalate_optical_x,
        lithium_tantalate_optical_y,
        lithium_tantalate_optical_z,
    )

    env = mw.Environment(wl=1.0)
    for optical_axis, material in [
        ("x", lithium_tantalate_optical_x),
        ("y", lithium_tantalate_optical_y),
        ("z", lithium_tantalate_optical_z),
    ]:
        for component in components:
            expected = 2.1432 if component == optical_axis else 2.1391
            assert np.isclose(material(env, component), expected)

    env = mw.Environment(wl=1.1)  # interpolated
    assert 2.1341 < lithium_tantalate_optical_z(env, "z").real < 2.1432
    assert 2.1305 < lithium_tantalate_optical_y(env, "x").real < 2.1391


def test_uniaxial_bad_axis():
    import pandas as pd

    df = pd.DataFrame({"wl": [1.0, 2.0], "nor": [1.0, 1.0], "ner": [2.0, 2.0]})
    with pytest.raises(ValueError):
        mw.SampledUniaxialMaterial.from_df("bad_uni", df, "w")  # ty: ignore[invalid-argument-type]


def test_anisotropic_sampled_material_axis_mapping():
    import pandas as pd

    import meow as mw

    df = pd.DataFrame(
        {"wl": [1.0, 2.0], "n1r": [1.0, 1.0], "n2r": [2.0, 2.0], "n3r": [3.0, 3.0]}
    )
    m = mw.SampledAnisotropicMaterial.from_df(
        "aniso_test", df, {1: "z", 2: "x", 3: "y"}
    )
    env = mw.Environment(wl=1.5)
    assert np.isclose(m(env, "z"), 1.0)
    assert np.isclose(m(env, "x"), 2.0)
    assert np.isclose(m(env, "y"), 3.0)
    with pytest.raises(ValueError):
        m(env)
    with pytest.raises(ValueError):
        mw.SampledAnisotropicMaterial.from_df("bad", df, {1: "x", 2: "x", 3: "y"})


def test_sampled_anisotropic_material_lithium_triborate():
    lithium_triborate = mw.materials.lithium_triborate
    env = mw.Environment(wl=1.0642)

    assert np.isclose(lithium_triborate(env, "x"), 1.5656)
    assert np.isclose(lithium_triborate(env, "y"), 1.5905)
    assert np.isclose(lithium_triborate(env, "z"), 1.6055)

    lithium_triborate_new_axes = mw.SampledAnisotropicMaterial.from_path(
        "lithium_triborate",
        axis_num_to_dir={2: "x", 3: "y", 1: "z"},
    )
    assert np.isclose(lithium_triborate_new_axes(env, "x"), 1.5905)
    assert np.isclose(lithium_triborate_new_axes(env, "y"), 1.6055)
    assert np.isclose(lithium_triborate_new_axes(env, "z"), 1.5656)


FUSED_SILICA_SELLMEIER_COEFFICIENTS = {
    "B": (0.6961663, 0.4079426, 0.8974794),
    "C": (0.0684043**2, 0.1162414**2, 9.896161**2),
}  # Malitson 1965, wavelength in um


def test_sellmeier_model():
    model = mw.SellmeierModel(**FUSED_SILICA_SELLMEIER_COEFFICIENTS)
    assert np.isclose(model(mw.Environment(wl=1.55)), 1.44402, atol=1e-5)
    assert np.isclose(model(mw.Environment(wl=0.5876)), 1.45846, atol=1e-5)
    # temperature is ignored
    assert np.isclose(
        model(mw.Environment(wl=1.55, T=0)), model(mw.Environment(wl=1.55, T=300))
    )
    # constant offset A
    model_a = mw.SellmeierModel(B=(), C=(), A=4.0)
    assert np.isclose(model_a(mw.Environment(wl=1.0)), 2.0)
    with pytest.raises(ValueError):
        mw.SellmeierModel(B=(1.0, 2.0), C=(1.0,))


def test_sellmeier_model_rejects_unphysical_wavelength():
    """n^2 < 0 lies outside the fit's range; Sellmeier cannot model it."""
    # fused silica just below its 9.896 um pole, where n^2 goes negative
    model = mw.SellmeierModel(**FUSED_SILICA_SELLMEIER_COEFFICIENTS)
    with pytest.raises(ValueError, match="not physical"):
        model(mw.Environment(wl=9.8))

    # indium phosphide fails outside its valid wavelength range
    with pytest.raises(ValueError, match="not physical"):
        mw.materials.indium_phosphide(mw.Environment(wl=0.6))
    with pytest.raises(ValueError, match="not physical"):
        mw.materials.indium_phosphide(mw.Environment(wl=30))
    # ... but fine in its physical band
    assert np.isclose(
        mw.materials.indium_phosphide(mw.Environment(wl=1.55)), 3.164929, atol=1e-5
    )


def test_dispersion_model_wavelength_range():
    """The base class rejects wavelengths outside the model's valid range."""
    model = mw.CauchyModel(
        A=1.5,
        B=0.01,
        wavelength_lower_limit=1.0,
        wavelength_upper_limit=2.0,
    )
    # inside the range (inclusive on both ends)
    assert np.isclose(model(mw.Environment(wl=1.0)), 1.5 + 0.01)
    assert np.isclose(model(mw.Environment(wl=2.0)), 1.5 + 0.01 / 4)
    # outside the range
    with pytest.raises(ValueError, match="outside the valid range"):
        model(mw.Environment(wl=0.99))
    with pytest.raises(ValueError, match="outside the valid range"):
        model(mw.Environment(wl=2.01))

    # the limits are inclusive and default to unbounded
    default = mw.CauchyModel(A=1.5, B=0.01)
    assert default(mw.Environment(wl=1e-3)) is not None
    with pytest.raises(ValueError, match="must not exceed"):
        mw.CauchyModel(
            A=1.5, B=0.01, wavelength_lower_limit=2.0, wavelength_upper_limit=1.0
        )


def test_cauchy_model():
    model = mw.CauchyModel(A=1.5, B=0.01, C=0.002)
    wl = 0.8
    assert np.isclose(model(mw.Environment(wl=wl)), 1.5 + 0.01 / wl**2 + 0.002 / wl**4)
    assert np.isclose(model(mw.Environment(wl=wl, T=0)), model(mw.Environment(wl=wl)))
    assert np.isclose(mw.CauchyModel(A=1.7)(mw.Environment(wl=1.3)), 1.7)


def test_dispersion_material_indium_phosphide():
    env_1550 = mw.Environment(wl=1.55)
    env_1310 = mw.Environment(wl=1.31)
    components_or_none: list[ComponentType | None] = [None, *components]
    for component in components_or_none:
        assert np.isclose(
            mw.materials.indium_phosphide(env_1550, component),
            3.164929,
            atol=1e-5,
        )
        assert np.isclose(
            mw.materials.indium_phosphide(env_1310, component),
            3.202006,
            atol=1e-5,
        )


def test_uniaxial_dispersion_material_aluminum_nitride():
    env_1550 = mw.Environment(wl=1.55)
    env_1310 = mw.Environment(wl=1.31)
    ordinary_components: list[ComponentType] = ["x", "y"]
    for component in ordinary_components:
        assert np.isclose(
            mw.materials.aluminum_nitride_optical_z(env_1550, component),
            2.119925,
            atol=1e-5,
        )
        assert np.isclose(
            mw.materials.aluminum_nitride_optical_z(env_1310, component),
            2.124379,
            atol=1e-5,
        )

    assert np.isclose(
        mw.materials.aluminum_nitride_optical_z(env_1550, "z"), 2.160207, atol=1e-5
    )
    assert np.isclose(
        mw.materials.aluminum_nitride_optical_z(env_1310, "z"), 2.165125, atol=1e-5
    )


def _ktp_index_from_paper(component: ComponentType, wavelength: float) -> float:
    """KTP refractive index from Eq. (1) of Kato and Takaoka (2002).

    Written in the form of the paper, n^2 = A + B1 / (wl^2 - C1) + B2 / (wl^2 - C2),
    with the wavelength in um.
    """
    wl2 = wavelength**2
    if component == "x":
        n_squared = 3.29100 + 0.04140 / (wl2 - 0.03978) + 9.35522 / (wl2 - 31.45571)
    elif component == "y":
        n_squared = 3.45018 + 0.04341 / (wl2 - 0.04597) + 16.98825 / (wl2 - 39.43799)
    elif component == "z":
        n_squared = 4.59423 + 0.06206 / (wl2 - 0.04763) + 110.80672 / (wl2 - 86.12171)
    else:
        msg = f"component must be 'x', 'y' or 'z'. Got {component!r}."
        raise ValueError(msg)
    return float(np.sqrt(n_squared))


def test_anisotropic_dispersion_material_ktp():
    ktp = mw.materials.ktp
    env = mw.Environment(wl=1.0642)

    # three different refractive indices, n_x < n_y < n_z
    n_x = ktp(env, "x")
    n_y = ktp(env, "y")
    n_z = ktp(env, "z")
    assert n_x.real < n_y.real < n_z.real

    # matches the Sellmeier equations of the paper, at the Nd:YAG wavelength and
    # at other wavelengths in the range of validity of the paper (0.43 - 3.54 um)
    for wavelength in [0.5, 0.8, 1.0642, 1.55, 3.0]:
        env_wl = mw.Environment(wl=wavelength)
        for component in components:
            assert np.isclose(
                ktp(env_wl, component),
                _ktp_index_from_paper(component, wavelength),
            )

    # the component is required
    with pytest.raises(ValueError):
        ktp(env)


def test_dispersion_material_serialization_roundtrip():
    mat = mw.UniaxialDispersionMaterial(
        name="disp_roundtrip",
        ordinary_dispersion_model=mw.CauchyModel(A=1.5, B=0.01),
        extraordinary_dispersion_model=mw.SellmeierModel(
            **FUSED_SILICA_SELLMEIER_COEFFICIENTS
        ),
        optical_axis="z",
    )
    mat2 = mw.UniaxialDispersionMaterial.model_validate(mat.model_dump())
    assert isinstance(mat2.extraordinary_dispersion_model, mw.SellmeierModel)
    env = mw.Environment(wl=1.2)
    for comp in components:
        assert np.isclose(mat(env, comp), mat2(env, comp))
