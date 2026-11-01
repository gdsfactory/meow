from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
import pytest

import meow as mw

show = False  # default to not showing if running from pytest

# small list of 3 modes to reuse
test_structures = [
    mw.Structure(
        geometry=mw.Box(
            x_min=-0.5, x_max=0.5, y_min=-0.5, y_max=0.5, z_min=-1, z_max=1
        ),
        material=mw.silicon,
    )
]
test_cell = mw.Cell(
    structures=test_structures,
    z_min=-1,
    z_max=1,
    mesh=mw.Mesh2D(
        x=np.linspace(-2, 2, 50),
        y=np.linspace(-2, 2, 50),
    ),
)
test_xsec = mw.CrossSection.from_cell(cell=test_cell, env=mw.Environment(wl=1.55, T=20))


def _compute_modes_for_test() -> list[mw.Mode]:
    return mw.compute_modes(cs=test_xsec, num_modes=3)


@pytest.fixture(scope="module")
def modes_for_test() -> list[mw.Mode]:
    """Computed once per module, only if a selected test requests it."""
    return _compute_modes_for_test()


@pytest.fixture(autouse=True)
def _close_figures() -> Iterator[None]:
    """Close all figures after each test so tests don't draw on each other's."""
    yield
    plt.close("all")


def test_cell_visualization_no_vacuum():
    """Test cell visualization when there are no vacuum grid points."""
    structures = [
        mw.Structure(
            material=mw.silicon,
            geometry=mw.Box(
                x_min=-0.225, x_max=+0.225, y_min=0, y_max=0.22, z_min=-1, z_max=+1
            ),
            mesh_order=1,
        ),
        mw.Structure(
            material=mw.silicon_oxide,
            geometry=mw.Box(x_min=-1, x_max=+1, y_min=-1, y_max=+1, z_min=-1, z_max=+1),
            mesh_order=2,
        ),
    ]

    cell = mw.Cell(
        structures=structures,
        z_min=0,
        z_max=0,
        mesh=mw.Mesh2D(x=np.linspace(-1, 1, 100), y=np.linspace(-1, 1, 100)),
    )

    _fig, ax = plt.subplots()
    mw.visualize(cell, show=show, ax=ax)
    ax.set_title("test_cell_visualization_no_vacuum")


def test_cell_visualization_zero_thickness():
    """Test cell visualization when there are structures with zero thickness."""
    structures = [
        mw.Structure(
            material=mw.silicon,
            geometry=mw.Box(
                x_min=-0.225, x_max=+0.225, y_min=0, y_max=0, z_min=-1, z_max=+1
            ),
            mesh_order=1,
        ),
        mw.Structure(
            material=mw.silicon_oxide,
            geometry=mw.Box(x_min=-1, x_max=+1, y_min=-1, y_max=0, z_min=-1, z_max=+1),
            mesh_order=2,
        ),
    ]

    cell = mw.Cell(
        structures=structures,
        z_min=0,
        z_max=0,
        mesh=mw.Mesh2D(x=np.linspace(-1, 1, 100), y=np.linspace(-1, 1, 100)),
    )

    _fig, ax = plt.subplots()
    mw.visualize(cell, show=show, ax=ax)
    ax.set_title("test_cell_visualization_zero_thickness")


def test_cross_section_visualization_anisotropic():
    """Test cross section visualization for anisotropic materials.

    Make a cross section with some anisotropic material (tidy3d LiNbO3)
    and test that visualize called with index_component='x' or 'y' or 'z'
    results in top of refractive index legend changing according to the component.
    """

    for optical_axis in ["x", "y", "z"]:
        ln = mw.TidyMaterial(
            name="LiNbO3",
            variant="Zelmon1997",
            optical_axis=optical_axis,
        )

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
            mw.visualize(
                xsec,
                index_component=index_component_vis,
                ax=ax,
                show=show,
            )
            ax.set_title(
                f"test_cross_section_visualization_anisotropic_{index_component_vis}"
            )

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


@pytest.mark.parametrize(
    (
        "fields",
        "draw_refractive_index",
        "draw_mode_contours",
        "draw_mode_image",
        "draw_structure_boundaries",
    ),
    [
        (["Ex"], True, True, True, True),
        (["Ex", "Ey"], False, True, False, True),
        (["Ex", "Hx", "Pz"], True, False, True, False),
    ],
)
def test_mode_visualization(
    modes_for_test: list[mw.Mode],
    fields: list[str],
    *,
    draw_refractive_index: bool,
    draw_mode_contours: bool,
    draw_mode_image: bool,
    draw_structure_boundaries: bool,
):
    """Test that mode visualization succeeds for various cases.

    The cases are various combinations of the following:
    - Fields ["Ex"], ["Ex", "Ey"], ["Ex", "Hx", "Pz"]
    - draw_refractive_index True/False
    - draw_mode_contours True/False
    - draw_mode_image True/False
    - draw_structure_boundaries True/False
    """

    mw.visualize(
        modes_for_test[0],
        fields=fields,
        draw_refractive_index=draw_refractive_index,
        draw_mode_contours=draw_mode_contours,
        draw_mode_image=draw_mode_image,
        draw_structure_boundaries=draw_structure_boundaries,
        show=show,
        title_prefix="test_mode_visualization ",
        mode_cmap="rainbow",
    )


@pytest.mark.parametrize(
    "fields",
    [
        (["Ex"]),
        (["Ex", "Ey"]),
        (["Ex", "Hx", "Pz"]),
    ],
)
def test_multiple_modes_visualization(modes_for_test: list[mw.Mode], fields: list[str]):
    """Test that multiple modes visualization succeeds for various arguments."""

    mw.visualize(
        modes_for_test,
        fields=fields,
        draw_refractive_index=True,
        draw_mode_contours=True,
        draw_mode_image=True,
        draw_structure_boundaries=True,
        show=show,
        title_prefix="test_multiple_modes_visualization ",
        mode_cmap="rainbow",
    )


def test_mode_draw_structure_boundaries():
    """Test drawing structures inside and outside the mesh and ensure they appear"""

    # color that doesn't appear elsewhere in the plot so we can
    # programmatically determine whether structure boundary is drawn
    structure_color = "#FF69B4"
    structure_color_rgb = np.array(
        [
            int(f"0x{structure_color[1:3]}", 16),
            int(f"0x{structure_color[3:5]}", 16),
            int(f"0x{structure_color[5:7]}", 16),
        ]
    )

    # make cell with structure fully, partially, or not at all in the mesh
    # then visualize with draw_structure_boundaries=True
    # and check whether structure_color appears in the image accordingly

    #### structure fully in mesh

    fig, ax = plt.subplots()
    cell = mw.Cell(
        structures=test_structures,
        z_min=-1,
        z_max=1,
        mesh=mw.Mesh2D(x=np.linspace(-1, 1, 50), y=np.linspace(-1, 1, 50)),
    )
    xsec = mw.CrossSection.from_cell(cell=cell, env=mw.Environment(wl=1.55, T=20))
    modes = mw.compute_modes(cs=xsec, num_modes=1)
    mw.visualize(
        modes[0],
        fields=["Ex"],
        draw_structure_boundaries=True,
        structure_boundaries_color=structure_color,
        ax=ax,
        show=False,
    )
    # check that structure_color appears in the image
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())  # ty: ignore[unresolved-attribute]

    if show:
        plt.show()

    assert np.any(np.all(img[:, :, :3] == structure_color_rgb, axis=-1))

    #### structure partially in mesh

    fig, ax = plt.subplots()
    cell = mw.Cell(
        structures=test_structures,
        z_min=-1,
        z_max=1,
        mesh=mw.Mesh2D(x=np.linspace(-2, 0, 50), y=np.linspace(-2, 0, 50)),
    )
    xsec = mw.CrossSection.from_cell(cell=cell, env=mw.Environment(wl=1.55, T=20))
    modes = mw.compute_modes(cs=xsec, num_modes=1)
    mw.visualize(
        modes[0],
        fields=["Ex"],
        draw_structure_boundaries=True,
        structure_boundaries_color=structure_color,
        ax=ax,
        show=False,
    )
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())  # ty: ignore[unresolved-attribute]

    if show:
        plt.show()

    assert np.any(np.all(img[:, :, :3] == structure_color_rgb, axis=-1))

    # test that plot boundaries are correct, as plotting structure boundaries
    # outside plot can mess things up if you're not careful
    # with some tolerance because the limits get changed to min and max of mesh
    assert np.isclose(ax.get_xlim()[0], -2, atol=0.05), (
        f"{ax.get_xlim()[0] = }, should be near -2"
    )
    assert np.isclose(ax.get_xlim()[1], 0, atol=0.05), (
        f"{ax.get_xlim()[1] = }, should be near 0"
    )
    assert np.isclose(ax.get_ylim()[0], -2, atol=0.05), (
        f"{ax.get_ylim()[0] = }, should be near -2"
    )
    assert np.isclose(ax.get_ylim()[1], 0, atol=0.05), (
        f"{ax.get_ylim()[1] = }, should be near 0"
    )

    ##### structure outside of mesh

    fig, ax = plt.subplots()
    cell = mw.Cell(
        structures=test_structures,
        z_min=-1,
        z_max=1,
        mesh=mw.Mesh2D(x=np.linspace(2, 3, 50), y=np.linspace(2, 3, 50)),
    )
    xsec = mw.CrossSection.from_cell(cell=cell, env=mw.Environment(wl=1.55, T=20))
    modes = mw.compute_modes(cs=xsec, num_modes=1)
    mw.visualize(
        modes[0],
        fields=["Ex"],
        draw_structure_boundaries=True,
        structure_boundaries_color=structure_color,
        ax=ax,
        show=False,
    )
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())  # ty: ignore[unresolved-attribute]

    if show:
        plt.show()

    assert not np.any(np.all(img[:, :, :3] == structure_color_rgb, axis=-1))

    # test that plot boundaries are correct, as plotting structure boundaries
    # outside plot can mess things up if you're not careful
    # with some tolerance because the limits get changed to min and max of mesh
    assert np.isclose(ax.get_xlim()[0], 2, atol=0.05), (
        f"{ax.get_xlim()[0] = }, should be near 2"
    )
    assert np.isclose(ax.get_xlim()[1], 3, atol=0.05), (
        f"{ax.get_xlim()[1] = }, should be near 3"
    )
    assert np.isclose(ax.get_ylim()[0], 2, atol=0.05), (
        f"{ax.get_ylim()[0] = }, should be near 2"
    )
    assert np.isclose(ax.get_ylim()[1], 3, atol=0.05), (
        f"{ax.get_ylim()[1] = }, should be near 3"
    )


if __name__ == "__main__":
    show = True  # show figures if running from command line
    modes = _compute_modes_for_test()

    # run tests with a subset of parameter values
    test_cell_visualization_no_vacuum()
    test_cell_visualization_zero_thickness()
    test_cross_section_visualization_anisotropic()
    test_mode_visualization(
        modes,
        fields=["Ex", "Hx", "Pz"],
        draw_refractive_index=True,
        draw_mode_contours=False,
        draw_mode_image=True,
        draw_structure_boundaries=False,
    )
    test_multiple_modes_visualization(modes, fields=["Ex", "Hx", "Pz"])
    test_mode_draw_structure_boundaries()
