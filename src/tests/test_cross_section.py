import numpy as np

import meow as mw


def test_anisotropic_material():
    """Test that a cross section made entirely of an anisotropic material returns
    appropriate nx, ny, nz values"""
    anisotropic_material = mw.AnisotropicIndexMaterial(
        name="test_anisotropic_material", nx=1.0, ny=2.0, nz=3.0
    )
    box = mw.Structure(
        material=anisotropic_material,
        geometry=mw.Box(x_min=0, x_max=1, y_min=0, y_max=1, z_min=0, z_max=1),
    )
    cell = mw.Cell(
        structures=[box],
        mesh=mw.Mesh2D(x=np.linspace(0, 1, 10), y=np.linspace(0, 1, 10)),
        z_min=0,
        z_max=1,
    )
    cross_section = mw.CrossSection.from_cell(cell=cell, env=mw.Environment())
    assert np.all(cross_section.structures[0].material.nx == 1.0)  # ty: ignore[unresolved-attribute]
    assert np.all(cross_section.structures[0].material.ny == 2.0)  # ty: ignore[unresolved-attribute]
    assert np.all(cross_section.structures[0].material.nz == 3.0)  # ty: ignore[unresolved-attribute]


def test_cross_section_from_cell():
    """Tests against a minor bug that previously existed

    _cell did not get assigned properly when constructing a CrossSection with from_cell.
    """
    cell = mw.Cell(
        structures=[
            mw.Structure(
                material=mw.silicon,
                geometry=mw.Box(x_min=0, x_max=1, y_min=0, y_max=1, z_min=0, z_max=1),
            )
        ],
        mesh=mw.Mesh2D(x=np.linspace(0, 1, 10), y=np.linspace(0, 1, 10)),
        z_min=0,
        z_max=1,
    )
    env = mw.Environment()
    xsec = mw.CrossSection.from_cell(cell=cell, env=env)
    assert xsec._cell is cell
    assert xsec.env is env
    assert xsec.structures == cell.structures_2d
    assert xsec.mesh is cell.mesh


def test_winner_takes_all_background_wins_minority_corner_pixel():
    """An interface pixel mostly outside a structure must get the background index.

    Mesh nodes are at x, y = 0, 1, 2, 3, 4, so the Ez pixel at (2, 2) has dual
    cell [1.5, 2.5] x [1.5, 2.5]. The box corner sits at (1.9, 1.9): the box
    overlaps the pixel's (smaller) rasterization cell, so the rasterizer labels
    the pixel "core", but the box covers only 0.6 * 0.6 = 36% of the dual cell. With
    winner-takes-all (< 50% coverage -> background) the pixel must be vacuum.
    """
    core = mw.IndexMaterial(name="core", n=2.0)
    box = mw.Structure(
        material=core,
        geometry=mw.Box(x_min=1.9, x_max=3.0, y_min=1.9, y_max=3.0, z_min=0, z_max=1),
    )
    cell = mw.Cell(
        structures=[box],
        mesh=mw.Mesh2D(x=np.linspace(0, 4, 5), y=np.linspace(0, 4, 5)),
        z_min=0,
        z_max=1,
    )
    xs = mw.CrossSection.from_cell(
        cell=cell, env=mw.Environment(), subpixel_smoothing=False
    )

    assert cell.mesh.Xz[2, 2] == 2.0
    assert cell.mesh.Yz[2, 2] == 2.0
    assert np.real(xs.nz[2, 2]) == 1.0  # background, not the core's 2.0
