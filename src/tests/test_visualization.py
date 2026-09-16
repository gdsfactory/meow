import numpy as np

import meow as mw

# Ensure that cell visualization does not crash with a few edge cases:
# no vacuum grid points, structures with zero thickness


def test_cell_visualization_no_vacuum():
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

    mw.visualize(cell, show=False)


def test_cell_visualization_zero_thickness():
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

    mw.visualize(cell, show=False)
