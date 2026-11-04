import numpy as np
import pytest

import meow as mw


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


def test_lithium_tantalate_optical_axes():
    from meow import materials as m

    env = mw.Environment(wl=1.0)
    for ax, mat in {
        "x": m.lithium_tantalate_optical_x,
        "y": m.lithium_tantalate_optical_y,
        "z": m.lithium_tantalate_optical_z,
    }.items():
        for comp in "xyz":
            expected = 2.1432 if comp == ax else 2.1391
            assert np.isclose(mat(env, comp), expected)
    env = mw.Environment(wl=1.1)  # interpolated
    assert 2.1305 < m.lithium_tantalate_optical_z(env, "z").real < 2.1432
    assert not hasattr(mw, "lithium_tantalate_optical_x")


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
    m = mw.AnisotropicSampledMaterial.from_df(
        "aniso_test", df, {1: "z", 2: "x", 3: "y"}
    )
    env = mw.Environment(wl=1.5)
    assert np.isclose(m(env, "z"), 1.0)
    assert np.isclose(m(env, "x"), 2.0)
    assert np.isclose(m(env, "y"), 3.0)
    with pytest.raises(ValueError):
        m(env)
    with pytest.raises(ValueError):
        mw.AnisotropicSampledMaterial.from_df("bad", df, {1: "x", 2: "x", 3: "y"})


def test_lithium_triborate():
    from meow import materials as m

    env = mw.Environment(wl=1.0642)
    assert np.isclose(m.lithium_triborate(env, "x"), 1.5656)
    assert np.isclose(m.lithium_triborate(env, "y"), 1.5905)
    assert np.isclose(m.lithium_triborate(env, "z"), 1.6055)


FUSED_SILICA = {
    "B": (0.6961663, 0.4079426, 0.8974794),
    "C": (0.0684043**2, 0.1162414**2, 9.896161**2),
}  # Malitson 1965, wavelength in um


def test_sellmeier_model():
    model = mw.SellmeierModel(**FUSED_SILICA)
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


def test_cauchy_model():
    model = mw.CauchyModel(A=1.5, B=0.01, C=0.002)
    wl = 0.8
    assert np.isclose(model(mw.Environment(wl=wl)), 1.5 + 0.01 / wl**2 + 0.002 / wl**4)
    assert np.isclose(model(mw.Environment(wl=wl, T=0)), model(mw.Environment(wl=wl)))
    assert np.isclose(mw.CauchyModel(A=1.7)(mw.Environment(wl=1.3)), 1.7)


def test_dispersion_model_is_abstract():
    with pytest.raises(TypeError):
        mw.DispersionModelBase()


def test_dispersion_material():
    model = mw.SellmeierModel(**FUSED_SILICA)
    mat = mw.DispersionMaterial(name="disp_fused_silica", dispersion_model=model)
    env = mw.Environment(wl=1.55)
    for comp in [None, "x", "y", "z"]:
        assert np.isclose(mat(env, comp), 1.44402, atol=1e-5)  # ty: ignore[invalid-argument-type]


def test_uniaxial_dispersion_material():
    ordinary = mw.CauchyModel(A=1.5, B=0.01)
    extraordinary = mw.SellmeierModel(**FUSED_SILICA)
    env = mw.Environment(wl=1.0)
    for optical_axis in "xyz":
        mat = mw.UniaxialDispersionMaterial(
            name=f"disp_uniaxial_{optical_axis}",
            ordinary_dispersion_model=ordinary,
            extraordinary_dispersion_model=extraordinary,
            optical_axis=optical_axis,
        )
        for comp in "xyz":
            expected = extraordinary(env) if comp == optical_axis else ordinary(env)
            assert np.isclose(mat(env, comp), expected)
        with pytest.raises(ValueError):
            mat(env)
    with pytest.raises(ValueError):
        mw.UniaxialDispersionMaterial(
            name="disp_uniaxial_bad",
            ordinary_dispersion_model=ordinary,
            extraordinary_dispersion_model=extraordinary,
            optical_axis="w",
        )


def test_biaxial_dispersion_material():
    models = {
        "x": mw.CauchyModel(A=1.4),
        "y": mw.CauchyModel(A=1.5, B=0.02),
        "z": mw.SellmeierModel(**FUSED_SILICA),
    }
    mat = mw.BiaxialDispersionMaterial(
        name="disp_biaxial",
        dispersion_model_x=models["x"],
        dispersion_model_y=models["y"],
        dispersion_model_z=models["z"],
    )
    env = mw.Environment(wl=1.0)
    assert np.isclose(mat(env, "x"), 1.4)
    assert np.isclose(mat(env, "y"), 1.52)
    assert np.isclose(mat(env, "z"), models["z"](env))
    with pytest.raises(ValueError):
        mat(env)


def test_dispersion_material_serialization_roundtrip():
    mat = mw.UniaxialDispersionMaterial(
        name="disp_roundtrip",
        ordinary_dispersion_model=mw.CauchyModel(A=1.5, B=0.01),
        extraordinary_dispersion_model=mw.SellmeierModel(**FUSED_SILICA),
        optical_axis="z",
    )
    mat2 = mw.UniaxialDispersionMaterial.model_validate(mat.model_dump())
    assert isinstance(mat2.extraordinary_dispersion_model, mw.SellmeierModel)
    env = mw.Environment(wl=1.2)
    for comp in "xyz":
        assert np.isclose(mat(env, comp), mat2(env, comp))
