import matplotlib.pyplot as plt
import numpy as np

import meow as mw
from meow import materials

# set to True (by running this file with -visualize) to show the plots
visualize = False


def _make_xsec(
    core: mw.Material,
    env: mw.Environment,
    cladding: mw.Material | None = None,
) -> mw.CrossSection:
    structures = []
    if cladding is not None:
        structures.append(
            mw.Structure(
                material=cladding,
                geometry=mw.Box(
                    x_min=-2, x_max=2, y_min=-2, y_max=2, z_min=-1, z_max=1
                ),
                mesh_order=2,
            )
        )
    structures.append(
        mw.Structure(
            material=core,
            geometry=mw.Box(
                x_min=-0.5, x_max=0.5, y_min=-0.5, y_max=0.5, z_min=-1, z_max=1
            ),
            mesh_order=1,
        )
    )
    cell = mw.Cell(
        structures=structures,
        z_min=0,
        z_max=0,
        mesh=mw.Mesh2D(x=np.linspace(-2, 2, 100), y=np.linspace(-2, 2, 100)),
    )
    return mw.CrossSection.from_cell(cell=cell, env=env)


def _colorbar_upper_limits(fig: plt.Figure) -> list[float]:
    return [
        float(collection.colorbar.get_ticks()[-1])
        for ax in fig.axes
        for collection in ax.collections
        if collection.colorbar is not None
    ]


def _check_colorbar_upper_limits(
    xsec: mw.CrossSection, expected: dict[str, float], label: str
) -> None:
    """Check the colorbar upper limit for each visualized index component."""
    for component, n_expected in expected.items():
        fig, ax = plt.subplots()
        ax.set_title(f"{label}: index_component={component}")
        mw.visualize(xsec, index_component=component, ax=ax, show=False)
        limits = _colorbar_upper_limits(fig)
        assert limits, "no colorbar found"
        for limit in limits:
            assert np.isclose(limit, n_expected, rtol=1e-3), (
                f"{label}: index component {component}: colorbar upper limit is "
                f"{limit}, but should be {n_expected}."
            )
        if visualize:
            plt.show()
        plt.close(fig)


def test_cross_section_visualization_anisotropic():
    # make a cross section with some anisotopic material (tidy3d LiNbO3)
    # and test that visualize called with index_component='x' or 'y' or 'z'
    # results in top of refractive index legend changing according to the component.

    for optical_axis in ["x", "y", "z"]:
        ln = {
            "x": materials.lithium_niobate_optical_x,
            "y": materials.lithium_niobate_optical_y,
            "z": materials.lithium_niobate_optical_z,
        }[optical_axis]

        structures = [
            mw.Structure(
                material=ln,
                geometry=mw.Box(
                    x_min=-0.5, x_max=0.5, y_min=-0.5, y_max=0.5, z_min=-1, z_max=1
                ),
                mesh_order=1,
            )
        ]

        cell = mw.Cell(
            structures=structures,
            z_min=0,
            z_max=0,
            mesh=mw.Mesh2D(
                x=np.linspace(-2, 2, 100),
                y=np.linspace(-2, 2, 100),
            ),
        )

        env = mw.Environment(wl=1.55, T=20)
        xsec = mw.CrossSection.from_cell(cell=cell, env=env)
        for index_component_vis in ["x", "y", "z"]:
            fig, ax = plt.subplots()
            mw.visualize(xsec, index_component=index_component_vis, ax=ax, show=False)
            if visualize:
                plt.show()

            # find  colorbar of the visualized cross section,
            # and verify that the upper limit is that of the
            # refractive index of LiNbO3 along index_component_vis
            for ax in fig.axes:
                for collection in ax.collections:
                    cbar = collection.colorbar
                    if cbar is not None:
                        ticks = cbar.get_ticks()

                        assert np.isclose(
                            ticks[-1],
                            ln(env, component=index_component_vis),  # ty: ignore[invalid-argument-type]
                        ), (
                            f"Optical axis = {optical_axis}, "
                            f"index component to visualize = {index_component_vis}."
                            f"Colorbar upper limit is {ticks[-1]}, "
                            f"but should be {ln(env, component=index_component_vis)}"  # ty: ignore[invalid-argument-type]
                        )
                        assert np.isclose(
                            ticks[-1],
                            2.14 if index_component_vis == optical_axis else 2.21,
                            rtol=0.01,
                        )


def test_cross_section_visualization_anisotropic_index_material():
    # constant anisotropic index (2, 3, 4) in an isotropic cladding
    core = mw.AnisotropicIndexMaterial(name="vis_aniso_234", nx=2, ny=3, nz=4)
    cladding = mw.IndexMaterial(name="vis_clad_1p5", n=1.5)
    env = mw.Environment(wl=1.55)
    xsec = _make_xsec(core, env, cladding)
    _check_colorbar_upper_limits(
        xsec, {"x": 2.0, "y": 3.0, "z": 4.0}, "AnisotropicIndexMaterial"
    )


def test_cross_section_visualization_anisotropic_sampled_uniaxial():
    # sampled uniaxial lithium tantalate, optical axis along x, y or z
    env = mw.Environment(wl=1.55)
    for optical_axis in ["x", "y", "z"]:
        lt = getattr(materials, f"lithium_tantalate_optical_{optical_axis}")
        xsec = _make_xsec(lt, env)
        # interpolated between 1.4 um and 1.6 um table entries (Bond 1965)
        expected = {comp: 2.1228 if comp == optical_axis else 2.11895 for comp in "xyz"}
        _check_colorbar_upper_limits(
            xsec, expected, f"LiTaO3 optical axis {optical_axis}"
        )


def test_cross_section_visualization_anisotropic_sampled_biaxial():
    # sampled biaxial lithium triborate: n_x < n_y < n_z, all distinct
    env = mw.Environment(wl=1.0642)
    xsec = _make_xsec(materials.lithium_triborate, env)
    _check_colorbar_upper_limits(
        xsec, {"x": 1.5656, "y": 1.5905, "z": 1.6055}, "LiB3O5"
    )


def test_cross_section_visualization_anisotropic_in_cladding():
    # anisotropic LiNbO3 core in a silicon oxide cladding: upper limit is the core
    env = mw.Environment(wl=1.55, T=20)
    core = materials.lithium_niobate_optical_z
    xsec = _make_xsec(core, env, materials.silicon_oxide)
    _check_colorbar_upper_limits(
        xsec,
        {c: float(core(env, c).real) for c in ["x", "y", "z"]},  # ty: ignore[invalid-argument-type]
        "LiNbO3 in SiO2",
    )


def test_cross_section_visualization_dispersion_material():
    # isotropic dispersion material: the same upper limit for every component
    sellmeier = mw.SellmeierModel(
        B=(0.6961663, 0.4079426, 0.8974794),
        C=(0.0684043**2, 0.1162414**2, 9.896161**2),
    )
    mat = mw.DispersionMaterial(name="vis_disp_sellmeier", dispersion_model=sellmeier)
    env = mw.Environment(wl=1.0)
    xsec = _make_xsec(mat, env)
    n = 1.45042
    _check_colorbar_upper_limits(
        xsec, {"x": n, "y": n, "z": n}, "DispersionMaterial (Sellmeier)"
    )


def test_cross_section_visualization_uniaxial_dispersion_material():
    # Cauchy ordinary (1.5 + 0.01/wl^2) and extraordinary (1.7 + 0.02/wl^2) indices
    env = mw.Environment(wl=1.0)
    for optical_axis in ["x", "y", "z"]:
        mat = mw.UniaxialDispersionMaterial(
            name=f"vis_disp_uniaxial_{optical_axis}",
            ordinary_dispersion_model=mw.CauchyModel(A=1.5, B=0.01),
            extraordinary_dispersion_model=mw.CauchyModel(A=1.7, B=0.02),
            optical_axis=optical_axis,
        )
        xsec = _make_xsec(mat, env, materials.silicon_oxide)
        expected = {c: 1.72 if c == optical_axis else 1.51 for c in "xyz"}
        _check_colorbar_upper_limits(
            xsec, expected, f"UniaxialDispersionMaterial optical axis {optical_axis}"
        )


def test_cross_section_visualization_biaxial_dispersion_material():
    env = mw.Environment(wl=1.0)
    mat = mw.BiaxialDispersionMaterial(
        name="vis_disp_biaxial",
        dispersion_model_x=mw.CauchyModel(A=1.6),
        dispersion_model_y=mw.CauchyModel(A=1.8, B=0.02),
        dispersion_model_z=mw.CauchyModel(A=2.0, B=0.04),
    )
    xsec = _make_xsec(mat, env, materials.silicon_oxide)
    _check_colorbar_upper_limits(
        xsec, {"x": 1.6, "y": 1.82, "z": 2.04}, "BiaxialDispersionMaterial"
    )


if __name__ == "__main__":
    # if run with command line argument -visualize, show the plots
    import sys

    visualize = "-visualize" in sys.argv
    test_cross_section_visualization_anisotropic()
    test_cross_section_visualization_anisotropic_index_material()
    test_cross_section_visualization_anisotropic_sampled_uniaxial()
    test_cross_section_visualization_anisotropic_sampled_biaxial()
    test_cross_section_visualization_anisotropic_in_cladding()
    test_cross_section_visualization_dispersion_material()
    test_cross_section_visualization_uniaxial_dispersion_material()
    test_cross_section_visualization_biaxial_dispersion_material()
