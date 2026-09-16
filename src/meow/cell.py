"""an EME Cell."""

from __future__ import annotations

from itertools import pairwise
from typing import Annotated, Any, overload

import numpy as np
import shapely
from pydantic import Field
from scipy.ndimage import binary_dilation
from shapely.ops import unary_union

from meow.arrays import Dim, DType, IntArray2D, NDArray
from meow.base_model import BaseModel, cached_property
from meow.materials import Material
from meow.mesh import Mesh2D
from meow.structures import Structure2D, Structure3D, sort_structures


class Cell(BaseModel):
    """An EME Cell.

    This defines an interval in z (the direction of propagation) within
    the simulation domain. The intersecting Structure3Ds are discretized by
    a given mesh at the center of the Cell

    """

    structures: list[Structure3D] = Field(
        description="the 3D structures which will be sliced by the cell"
    )
    mesh: Mesh2D = Field(description="the mesh to discretize the structures with")
    z_min: float = Field(description="the starting z-coordinate of the cell")
    z_max: float = Field(description="the ending z-coordinate of the cell")

    @property
    def z(self) -> float:
        """The z-position of the center of the cell."""
        return 0.5 * (self.z_min + self.z_max)

    @property
    def length(self) -> float:
        """The length of the cell."""
        return np.abs(self.z_max - self.z_min)

    @cached_property
    def materials(self) -> dict[Material, int]:
        """A mapping of the materials in the cell to their indices."""
        materials = {}
        for i, structure in enumerate(sort_structures(self.structures), start=1):
            if structure.material not in materials:
                materials[structure.material] = i
        return materials

    @cached_property
    def structures_2d(self) -> list[Structure2D]:
        """The 2D structures in the cell."""
        z = 0.5 * (self.z_min + self.z_max)
        list_of_list = [s._project(z) for s in self.structures]
        structures = [s for ss in list_of_list for s in ss]
        return structures

    @cached_property
    def m_full(self) -> IntArray2D:
        """The full material mask for the cell."""
        return _create_full_material_array(
            mesh=self.mesh,
            structures=self.structures_2d,
            materials=self.materials,
        )

    def _visualize(
        self,
        *,
        ax: Any = None,
        cbar: bool = True,
        show: bool = True,
        **_: Any,
    ) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, to_rgba
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        colors = [(0, 0, 0, 0)] + [
            to_rgba(m.meta.get("color", (0, 0, 0, 0))) for m in self.materials
        ]
        cmap = ListedColormap(colors=colors)
        if ax is not None:
            plt.sca(ax)
        else:
            ax = plt.gca()
        plt.pcolormesh(
            self.mesh.X_full,
            self.mesh.Y_full,
            self.m_full,
            cmap=cmap,
            vmin=0,
            vmax=len(self.materials) + 1,
        )
        plt.axis("scaled")
        plt.grid(visible=True)
        if cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            labels = ["vacuum"]
            ticks = [0.5]
            for material, index in self.materials.items():
                labels.append(material.name)
                ticks.append(index + 0.5)

            _cbar = plt.colorbar(
                ticks=ticks,
                cax=cax,
            )
            _cbar.ax.set_yticklabels(labels, rotation=90, va="center")
            plt.sca(ax)
        if show:
            plt.show()


def create_cells(
    structures: list[Structure3D],
    mesh: Mesh2D | list[Mesh2D],
    Ls: Annotated[NDArray, Dim(1), DType("float64")],
    z_min: float = 0.0,
) -> list[Cell]:
    """Create multiple `Cell` objects with a `Mesh` and a collection of cell lengths.

    Args:
        structures: the 3D structures shared by all cells.
        mesh: a single mesh or a list of meshes, one per cell.
        Ls: a 1D array of cell lengths.
        z_min: the starting z-coordinate for the first cell.

    Returns:
        A list of Cell objects spanning the given lengths.
    """
    Ls = np.asarray(Ls, float)
    if Ls.ndim != 1:
        msg = f"Ls should be 1D. Got shape: {Ls.shape}."
        raise ValueError(msg)
    if Ls.shape[0] < 0:
        msg = f"length of Ls array should be nonzero. Got: {Ls}."
        raise ValueError(msg)

    meshes = [mesh] * Ls.shape[0] if isinstance(mesh, Mesh2D) else mesh
    if len(Ls) != len(meshes):
        msg = (
            "Number of meshes should correspond to number of lengths (length of Ls). "
            f"Got {len(meshes)} != {len(Ls)}."
        )
        raise ValueError(msg)

    z = np.cumsum(np.concatenate([np.asarray([z_min], float), Ls]))
    cells = [
        Cell(
            structures=structures,
            mesh=mesh,
            z_min=z_min,
            z_max=z_max,
        )
        for mesh, (z_min, z_max) in zip(meshes, pairwise(z), strict=False)
    ]

    return cells


def _create_full_material_array(
    mesh: Mesh2D,
    structures: list[Structure2D],
    materials: dict[Material, int],
) -> IntArray2D:
    m_full = np.zeros_like(mesh.X_full, dtype=np.int_)
    structures_dict = _classify_structures_by_mesh_order_and_material(
        structures, materials
    )
    for structs in structures_dict.values():
        _m_full = _rasterize_structure_group(mesh, structs, materials)
        mask = _m_full > 0
        m_full[mask] = _m_full[mask]

    return m_full


def _compute_pixel_cell_bounds(
    mesh: Mesh2D,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Compute (x_lo, x_hi) and (y_lo, y_hi) for each position in the full grid."""
    x = mesh.x_full
    y = mesh.y_full

    x_mid = 0.5 * (x[:-1] + x[1:])
    x_lo = np.empty_like(x)
    x_hi = np.empty_like(x)
    # Clamp edge dual-cell bounds to the simulation domain.
    x_lo[0] = x[0]
    x_lo[1:] = x_mid
    x_hi[-1] = x[-1]
    x_hi[:-1] = x_mid

    y_mid = 0.5 * (y[:-1] + y[1:])
    y_lo = np.empty_like(y)
    y_hi = np.empty_like(y)
    y_lo[0] = y[0]
    y_lo[1:] = y_mid
    y_hi[-1] = y[-1]
    y_hi[:-1] = y_mid

    return (x_lo, x_hi), (y_lo, y_hi)


def _rasterize_structure_group(
    mesh: Mesh2D,
    structures: list[Structure2D],
    materials: dict[Material, int],
) -> IntArray2D:
    """Rasterize a group of structures with the same (mesh_order, material).

    Uses point-in-polygon for interior pixels, then dilates by 1 pixel and
    tests pixel-cell intersection with the polygon to fill boundary gaps.
    """
    mat_idx = materials[structures[0].material]
    m_full = np.zeros_like(mesh.X_full, dtype=np.int_)

    # Step 1: point-in-polygon mask (existing _mask logic)
    combined_mask = np.zeros_like(mesh.X_full, dtype=bool)
    for structure in structures:
        combined_mask |= structure.geometry._mask(mesh.X_full, mesh.Y_full)
    m_full[combined_mask] = mat_idx

    # Step 2: dilate by 1 pixel to find candidate boundary pixels
    dilated = binary_dilation(combined_mask, structure=np.ones((3, 3)))
    candidates = dilated & ~combined_mask

    if not np.any(candidates):
        return m_full

    # Step 3: vectorized shapely intersection test on candidate pixels
    (x_lo, x_hi), (y_lo, y_hi) = _compute_pixel_cell_bounds(mesh)
    poly = unary_union([s.geometry._shapely_polygon() for s in structures])

    ci, cj = np.where(candidates)
    pixel_boxes = shapely.box(x_lo[ci], y_lo[cj], x_hi[ci], y_hi[cj])
    # Do not fill pixels that only touch the polygon boundary with zero area.
    overlap_area = shapely.area(shapely.intersection(poly, pixel_boxes))
    hits = overlap_area > 0.0
    m_full[ci[hits], cj[hits]] = mat_idx

    return m_full


@overload
def _classify_structures_by_mesh_order_and_material(
    structures: list[Structure3D], materials: dict[Material, int]
) -> dict[tuple[int, int], list[Structure3D]]: ...


@overload
def _classify_structures_by_mesh_order_and_material(
    structures: list[Structure2D], materials: dict[Material, int]
) -> dict[tuple[int, int], list[Structure2D]]: ...


def _classify_structures_by_mesh_order_and_material(
    structures: list[Structure3D] | list[Structure2D],
    materials: dict[Material, int],
) -> (
    dict[tuple[int, int], list[Structure2D]] | dict[tuple[int, int], list[Structure3D]]
):
    structures = sort_structures(structures)
    structures_dict = {}
    for structure in structures:
        mo = structure.mesh_order
        mat = materials[structure.material]
        if (mo, mat) not in structures_dict:
            structures_dict[mo, mat] = []
        structures_dict[mo, mat].append(structure)
    return structures_dict
