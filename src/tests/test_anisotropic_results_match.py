import logging
from pathlib import Path

import numpy as np
import pandas as pd

import meow as mw

logger = logging.getLogger(__name__)

env = mw.Environment(T=20, wl=1.55)
nx = 2
ny = 3
nz = 4
_test_anisotropic_material = mw.AnisotropicIndexMaterial(
    name="test_anisotropic_material",
    nx=nx,
    ny=ny,
    nz=nz,
)

waveguide_width = 0.5
waveguide_height = 0.3
waveguide_gap = 0.3

dl = 0.02

visualize = False


def test_anisotropic_evanescent_coupler_matches_tidy3d():
    # load tidy3d results
    tidy3d_results_fname = (
        Path(__file__).resolve().parent
        / "assets"
        / f"tidy3d_anisotropic_evanescent_coupler_data_{dl=}.csv"
    )
    tidy3d_results = pd.read_csv(tidy3d_results_fname)

    # construct simulation that matches tidy3d example

    x_min = -3 * waveguide_width
    x_max = +3 * waveguide_width
    y_min = -waveguide_height / 2 - 2 * waveguide_width
    y_max = +waveguide_height / 2 + 2 * waveguide_width

    structures = [
        mw.Structure(  # left waveguide
            material=_test_anisotropic_material,
            geometry=mw.Box(
                x_min=-waveguide_width - waveguide_gap / 2,
                x_max=-waveguide_gap / 2,
                y_min=-waveguide_height / 2,
                y_max=waveguide_height / 2,
                z_min=0,
                z_max=2,
            ),
            mesh_order=1,
        ),
        mw.Structure(  # right waveguide
            material=_test_anisotropic_material,
            geometry=mw.Box(
                x_min=waveguide_gap / 2,
                x_max=waveguide_gap / 2 + waveguide_width,
                y_min=-waveguide_height / 2,
                y_max=waveguide_height / 2,
                z_min=1,
                z_max=3,
            ),
            mesh_order=1,
        ),
    ]

    mesh = mw.Mesh2D(
        x=np.arange(x_min, x_max, dl),
        y=np.arange(y_min, y_max, dl),
        num_pml=(10, 10),
    )

    cells = mw.create_cells(
        structures=structures,
        mesh=mesh,
        Ls=np.array([1, 1, 1]),
        z_min=0,
    )

    xsecs = [mw.CrossSection.from_cell(cell=cell, env=env) for cell in cells]
    num_modes = [2, 4, 2]  # number of modes to compute for each cross-section
    modes = [
        mw.compute_modes(cs=xsec, num_modes=num_modes[i])
        for i, xsec in enumerate(xsecs)
    ]

    S_matrix, port_map = mw.compute_s_matrix(modes=modes, cells=cells)

    interaction_length_arr = np.linspace(0, 50, 100)
    cell_lengths_arr = [[1, length, 1] for length in interaction_length_arr]
    mode_pairs = [
        (0, 0),
    ]
    mode_pair_to_coupling = {pair: [] for pair in mode_pairs}

    for cell_lengths in cell_lengths_arr:
        S_matrix, _port_map = mw.compute_s_matrix(
            modes=modes, cell_lengths=cell_lengths
        )
        for pair in mode_pairs:
            mode_pair_to_coupling[pair].append(
                np.abs(
                    S_matrix[port_map[f"left@{pair[0]}"]][port_map[f"right@{pair[1]}"]]
                )
                ** 2
            )

    mode_pair_to_results_diff = {pair: [] for pair in mode_pairs}
    for mode_pair in mode_pairs:
        mode_pair_to_results_diff[mode_pair] = np.abs(
            np.array(mode_pair_to_coupling[mode_pair])
            - tidy3d_results[f"coupling_{mode_pair[0]}{mode_pair[1]}"]
        )

    max_diff = max([np.max(mode_pair_to_results_diff[pair]) for pair in mode_pairs])
    logger.info("Max difference between MEOW and Tidy3D: %.5f", max_diff)

    if visualize:
        from matplotlib import pyplot as plt

        _fig, ax = plt.subplots(layout="constrained", dpi=300)

        for mode_pair in mode_pairs:
            ax.plot(
                tidy3d_results.length,
                tidy3d_results[f"coupling_{mode_pair[0]}{mode_pair[1]}"],
                label=f"Coupling (Tidy3D) {mode_pair[0]}->{mode_pair[1]}",
                linestyle="dotted",
                linewidth=2,
            )
            ax.plot(
                interaction_length_arr,
                mode_pair_to_coupling[mode_pair],
                label=f"Coupling (MEOW) {mode_pair[0]}->{mode_pair[1]}",
                linestyle="solid",
                linewidth=0.5,
            )
        ax.legend()
        ax.set_xlabel("Interaction length (um)")
        ax.set_ylabel("Coupling")
        ax.set_title(f"Coupling vs interaction length. Max diff {max_diff:.5f}")
        plt.show()

    if max_diff > 0.01:
        error_msg = (
            f"MEOW results do not closely match "
            f"Tidy3D results. Max difference: {max_diff}"
        )
        raise AssertionError(error_msg)


if __name__ == "__main__":
    # if run with command line argument -visualize,
    # visualize results. Else, time all results
    import sys

    visualize = "-visualize" in sys.argv
    if visualize:
        test_anisotropic_evanescent_coupler_matches_tidy3d()
    else:
        # time all tests individually
        import time

        start = time.time()
        test_anisotropic_evanescent_coupler_matches_tidy3d()
        logger.info(
            "test_anisotropic_evanescent_coupler_matches_tidy3d took %.2f seconds",
            (time.time() - start),
        )
