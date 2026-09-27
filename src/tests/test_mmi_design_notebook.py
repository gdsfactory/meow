from pathlib import Path
from typing import Any

import nbformat
import numpy as np

NOTEBOOK = Path(__file__).parents[2] / "notebooks" / "06_mmi_1x2_design.ipynb"


def test_mmi_design_notebook_contains_full_recomputed_sweep() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)

    required = (
        "# Design a 1x2 MMI",
        "gf.components.mmi1x2",
        "def solve_mmi",
        "for width_index, width_mmi in enumerate(widths_mmi)",
        "for length_index, length_mmi in enumerate(lengths_mmi)",
        '"T_top"',
        '"T_bottom"',
        '"T_total"',
        '"insertion_loss_db"',
    )
    assert all(text in source for text in required)


def test_mmi_output_ports_are_localized_orthonormal_and_labeled() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    namespace: dict = {}
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        if "def solve_mmi" in cell.source:
            break
        if "nominal_component =" in cell.source:
            continue
        exec(  # noqa: S102 - execute trusted, version-controlled notebook cells
            compile(cell.source, f"notebook-cell-{index}", "exec"), namespace
        )

    component = namespace["build_mmi"](width_mmi=3.0, length_mmi=5.5)
    structures = namespace["extrude_mmi"](component)
    cells = namespace["make_cells"](structures, length_mmi=5.5)
    mw = namespace["mw"]
    cross_section = mw.CrossSection.from_cell(cell=cells[-1], env=namespace["ENV"])
    raw_modes = namespace["guided_te_modes"](cross_section)[:2]

    # Mimic an arbitrary real rotation of a nearly degenerate eigenspace.
    angle = 0.55
    rotated_modes = [
        np.cos(angle) * raw_modes[0] + np.sin(angle) * raw_modes[1],
        -np.sin(angle) * raw_modes[0] + np.cos(angle) * raw_modes[1],
    ]
    top, bottom = namespace["localize_output_modes"](rotated_modes)

    def centroid_and_positive_fraction(mode: Any) -> tuple[float, float]:
        density = mw.electric_energy_density(mode)
        x = np.asarray(mode.mesh.Xx)
        total = density.sum()
        centroid = float((x * density).sum() / total)
        positive_fraction = float(density[x > 0].sum() / total)
        return centroid, positive_fraction

    top_centroid, top_positive_fraction = centroid_and_positive_fraction(top)
    bottom_centroid, bottom_positive_fraction = centroid_and_positive_fraction(bottom)
    gram = np.array(
        [[mw.inner_product(a, b) for b in (top, bottom)] for a in (top, bottom)]
    )

    assert top_centroid > 0.25
    assert bottom_centroid < -0.25
    assert top_positive_fraction > 0.9
    assert bottom_positive_fraction < 0.1
    np.testing.assert_allclose(gram, np.eye(2), atol=1e-6)
