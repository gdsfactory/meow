"""A CrossSection."""

from __future__ import annotations

from typing import Any, Self

import numpy as np
import shapely
from pydantic import Field, PrivateAttr
from shapely.ops import unary_union

from meow.arrays import ComplexArray2D, IntArray2D
from meow.base_model import BaseModel, cached_property
from meow.cell import (
    Cell,
    _classify_structures_by_mesh_order_and_material,
    _create_full_material_array,
    sort_structures,
)
from meow.environment import Environment
from meow.materials import ComponentType, Material
from meow.mesh import Mesh2D
from meow.structures import Structure2D


class CrossSection(BaseModel):
    """A `CrossSection` is built from a `Cell` with an `Environment`.

    This uniquely defines the refractive index everywhere.
    """

    structures: list[Structure2D] = Field(
        description="the 2D structures in the CrossSection"
    )
    mesh: Mesh2D = Field(description="the mesh to discretize the structures with")
    env: Environment = Field(
        description="the environment for which the cross section was calculated"
    )
    subpixel_smoothing: bool = Field(
        default=True,
        description="use subpixel smoothing at interfaces; if False: winner-takes-all",
    )
    _cell: Cell | None = PrivateAttr(default=None)
    _n_full_cache: dict[ComponentType, ComplexArray2D] = PrivateAttr(
        default_factory=dict
    )

    @classmethod
    def from_cell(
        cls,
        *,
        cell: Cell,
        env: Environment,
        subpixel_smoothing: bool = True,
    ) -> Self:
        """Create a CrossSection from a Cell and Environment."""
        result = cls(
            structures=cell.structures_2d,
            mesh=cell.mesh,
            env=env,
            subpixel_smoothing=subpixel_smoothing,
        )
        result._cell = cell
        return result

    @cached_property
    def materials(self) -> dict[Material, int]:
        """Return a dictionary mapping materials to their indices."""
        materials: dict[Material, int] = {}
        for i, structure in enumerate(sort_structures(self.structures), start=1):
            if structure.material not in materials:
                materials[structure.material] = i
        return materials

    @cached_property
    def _m_full(self) -> IntArray2D:
        """Return the material index array for the full mesh."""
        return _create_full_material_array(self.mesh, self.structures, self.materials)

    def n_full(self, component: ComponentType = "x") -> ComplexArray2D:
        """Return the refractive index (along the given axis) for the full mesh."""
        # check if n_full for this component exists in cache
        if component in self._n_full_cache:
            return self._n_full_cache[component]
        n_full = np.ones_like(self.mesh.X_full, dtype=np.complex128)
        for material, idx in self.materials.items():
            n_full = np.where(
                self._m_full == idx, material(self.env, component=component), n_full
            )
        self._n_full_cache[component] = n_full
        return n_full

    @cached_property
    def nx(self) -> ComplexArray2D:
        """Return smoothed refractive index on the Ex positions.

        Associated with x axis (for anisotropic materials).
        """
        if self.subpixel_smoothing:
            return _compute_smoothed_n(
                self.mesh, self._m_full, self.materials, self.env, self.structures, "x"
            )
        return _compute_winner_takes_all_n(
            self.mesh, self._m_full, self.materials, self.env, self.structures, "x"
        )

    @cached_property
    def ny(self) -> ComplexArray2D:
        """Return smoothed refractive index on the Ey positions.

        Associated with y axis (for anisotropic materials).
        """
        if self.subpixel_smoothing:
            return _compute_smoothed_n(
                self.mesh, self._m_full, self.materials, self.env, self.structures, "y"
            )
        return _compute_winner_takes_all_n(
            self.mesh, self._m_full, self.materials, self.env, self.structures, "y"
        )

    @cached_property
    def nz(self) -> ComplexArray2D:
        """Return smoothed refractive index on the Ez positions.

        Associated with z axis (for anisotropic materials).
        """
        if self.subpixel_smoothing:
            return _compute_smoothed_n(
                self.mesh, self._m_full, self.materials, self.env, self.structures, "z"
            )
        return _compute_winner_takes_all_n(
            self.mesh, self._m_full, self.materials, self.env, self.structures, "z"
        )

    def _visualize(
        self,
        *,
        ax: Any = None,
        n_cmap: Any = None,
        cbar: bool = True,
        show: bool = True,
        index_component: ComponentType = "x",
        **ignored: Any,
    ) -> None:
        import matplotlib.pyplot as plt  # fmt: skip
        from matplotlib import colors  # fmt: skip
        from mpl_toolkits.axes_grid1 import make_axes_locatable  # fmt: skip

        debug_grid = ignored.pop("debug_grid", False)
        if n_cmap is None:
            n_cmap = colors.LinearSegmentedColormap.from_list(
                name="c_cmap", colors=["#ffffff", "#86b5dc"]
            )
        if ax is not None:
            plt.sca(ax)
        else:
            ax = plt.gca()
        n_full = np.real(self.n_full(component=index_component)).copy()
        n_full[0, 0] = 1.0
        plt.pcolormesh(self.mesh.X_full, self.mesh.Y_full, n_full, cmap=n_cmap)
        plt.axis("scaled")
        if not debug_grid:
            plt.grid(visible=True)
        else:
            dx = self.mesh.dx
            dy = self.mesh.dy
            x_ticks = np.sort(np.unique(self.mesh.X_full.ravel()))[::2]
            y_ticks = np.sort(np.unique(self.mesh.Y_full.ravel()))[::2]
            plt.xticks(x_ticks - 0.25 * dx, ["" for _ in x_ticks - 0.25 * dx])
            plt.yticks(y_ticks - 0.25 * dy, ["" for _ in y_ticks - 0.25 * dy])
            plt.xticks(
                x_ticks + 0.25 * dx, ["" for _ in x_ticks + 0.25 * dx], minor=True
            )
            plt.yticks(
                y_ticks + 0.25 * dy, ["" for _ in y_ticks + 0.25 * dy], minor=True
            )
            plt.grid(visible=True, which="major", ls="-")
            plt.grid(visible=True, which="minor", ls=":")
        if cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            values = np.unique(np.real(n_full))
            _cbar = plt.colorbar(ticks=values, cax=cax)
            # material_names = ['vacuum'] + [mat.name for mat in self.cell.materials]
            # labels = [f"\n{n}\n{v:.3f}" for n, v in zip(material_names, values)]
            labels = [f"{v:.3f}" for v in values]
            _cbar.ax.set_yticklabels(labels, rotation=90, va="center", ha="center")
            plt.sca(ax)
        if show:
            plt.show()


# --- Subpixel permittivity smoothing ---

_COMPONENT_SLICES = {
    "x": (slice(1, None, 2), slice(None, None, 2)),  # Ex: [1::2, ::2]
    "y": (slice(None, None, 2), slice(1, None, 2)),  # Ey: [::2, 1::2]
    "z": (slice(None, None, 2), slice(None, None, 2)),  # Ez: [::2, ::2]
}


def _dual_cell_bounds(
    mesh: Mesh2D,
    component: ComponentType,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute (x_lo, x_hi, y_lo, y_hi) for the dual cells of a field component.

    Each field component lives at a specific position on the Yee grid. Its dual
    cell is the area over which the effective permittivity should be averaged.

    Full-grid indices (in x_full/y_full):
      Ex at (2i+1, 2j):
        dual cell x=[x_full[2i], x_full[2i+2]], y=[y_full[2j-1], y_full[2j+1]]
      Ey at (2i, 2j+1):
        dual cell x=[x_full[2i-1], x_full[2i+1]], y=[y_full[2j], y_full[2j+2]]
      Ez at (2i, 2j):
        dual cell x=[x_full[2i-1], x_full[2i+1]], y=[y_full[2j-1], y_full[2j+1]]
    """
    xf = mesh.x_full
    yf = mesh.y_full

    si, sj = _COMPONENT_SLICES[component]
    # Number of component grid points
    len(range(*si.indices(len(xf))))
    len(range(*sj.indices(len(yf))))

    # Full-grid indices for this component
    fi = np.arange(len(xf))[si]  # shape (ni,)
    fj = np.arange(len(yf))[sj]  # shape (nj,)

    def _bounds(vals: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lo = np.empty(len(indices))
        hi = np.empty(len(indices))
        for k, idx in enumerate(indices):
            if idx > 0:
                lo[k] = vals[idx - 1]
            else:
                lo[k] = vals[0]
            if idx < len(vals) - 1:
                hi[k] = vals[idx + 1]
            else:
                hi[k] = vals[-1]
        return lo, hi

    x_lo, x_hi = _bounds(xf, fi)
    y_lo, y_hi = _bounds(yf, fj)

    return x_lo, x_hi, y_lo, y_hi


def _compute_smoothed_n(  # noqa: PLR0915
    mesh: Mesh2D,
    m_full: IntArray2D,
    materials: dict[Material, int],
    env: Environment,
    structures: list[Structure2D],
    component: ComponentType,
) -> ComplexArray2D:
    """Compute subpixel-smoothed refractive index for a field component.

    At interface pixels, the effective permittivity is computed using
    area-fraction-weighted averaging:
      - E parallel to interface:      arithmetic avg  eps_eff = sum(f_i * eps_i)
      - E perpendicular to interface: harmonic avg    1/eps_eff = sum(f_i / eps_i)
    """
    si, sj = _COMPONENT_SLICES[component]
    m_comp = m_full[si, sj]

    # Build eps array from material indices
    env_eps = np.complex128(1.0) ** 2  # background: vacuum (n=1)
    eps = np.full_like(m_comp, env_eps, dtype=np.complex128)
    mat_eps: dict[int, complex] = {0: env_eps}
    for material, idx in materials.items():
        n_val = material(env, component=component)
        e = n_val**2
        eps[m_comp == idx] = e
        mat_eps[idx] = e

    # Identify interface pixels (where material differs from any 4-neighbor)
    padded = np.pad(m_comp, 1, mode="edge")
    is_interface = (
        (m_comp != padded[:-2, 1:-1])  # top
        | (m_comp != padded[2:, 1:-1])  # bottom
        | (m_comp != padded[1:-1, :-2])  # left
        | (m_comp != padded[1:-1, 2:])  # right
    )

    if not np.any(is_interface):
        return np.sqrt(eps)

    # Compute dual cell bounds for this component
    x_lo, x_hi, y_lo, y_hi = _dual_cell_bounds(mesh, component)

    # Get interface pixel indices
    ii, jj = np.where(is_interface)

    # Build dual cell boxes (vectorized)
    dual_boxes = shapely.box(x_lo[ii], y_lo[jj], x_hi[ii], y_hi[jj])
    dual_areas = shapely.area(dual_boxes)

    # Compute area fractions per material for each interface pixel using
    # effective (non-overlapping) polygons that respect mesh-order overwrite.
    effective_polys = _effective_material_polygons(structures, materials)

    # Area fractions: shape (n_interface_pixels,) per material, guaranteed
    # to be non-overlapping by construction.
    fractions: dict[int, np.ndarray] = {}
    total_struct_fraction = np.zeros(len(ii), dtype=float)
    for idx, poly in effective_polys.items():
        intersections = shapely.intersection(poly, dual_boxes)
        areas = shapely.area(intersections)
        frac = areas / dual_areas
        fractions[idx] = frac
        total_struct_fraction += frac

    # Clip accumulated area to avoid tiny numerical overshoots > 1.
    total_struct_fraction = np.clip(total_struct_fraction, 0.0, 1.0)

    # Background (vacuum) gets the remainder.
    fractions[0] = np.maximum(1.0 - total_struct_fraction, 0.0)

    # Determine interface orientation
    # Use padded array to check material changes in x and y directions
    diff_x = padded[:-2, 1:-1][ii, jj] != padded[2:, 1:-1][ii, jj]
    diff_y = padded[1:-1, :-2][ii, jj] != padded[1:-1, 2:][ii, jj]

    normal_x = diff_x & ~diff_y  # vertical interface, normal along x
    normal_y = ~diff_x & diff_y  # horizontal interface, normal along y
    # ~normal_x & ~normal_y  # corner or ambiguous -> arithmetic avg

    # Determine which averaging to use for this component at each pixel
    # harmonic when E is perpendicular to interface (E component == normal direction)
    # arithmetic when E is parallel to interface
    use_harmonic = np.zeros(len(ii), dtype=bool)
    if component == "x":
        use_harmonic[normal_x] = True  # Ex perpendicular to vertical interface
    elif component == "y":
        use_harmonic[normal_y] = True  # Ey perpendicular to horizontal interface
    # Ez is always parallel to interface -> always arithmetic
    # Corners -> arithmetic (use_harmonic stays False)

    # Compute effective eps for each interface pixel
    eps_eff = np.zeros(len(ii), dtype=np.complex128)

    # Arithmetic average: eps_eff = sum(f_i * eps_i)
    arith_mask = ~use_harmonic
    if np.any(arith_mask):
        eps_arith = np.zeros(arith_mask.sum(), dtype=np.complex128)
        for idx, frac in fractions.items():
            eps_arith += frac[arith_mask] * mat_eps[idx]
        eps_eff[arith_mask] = eps_arith

    # Harmonic average: 1/eps_eff = sum(f_i / eps_i)
    if np.any(use_harmonic):
        inv_eps_harm = np.zeros(use_harmonic.sum(), dtype=np.complex128)
        for idx, frac in fractions.items():
            e = mat_eps[idx]
            if abs(e) > 0:
                inv_eps_harm += frac[use_harmonic] / e
        # Avoid division by zero
        safe = np.abs(inv_eps_harm) > 1e-30
        eps_harm = np.zeros_like(inv_eps_harm)
        eps_harm[safe] = 1.0 / inv_eps_harm[safe]
        eps_harm[~safe] = eps[ii[use_harmonic], jj[use_harmonic]][~safe]
        eps_eff[use_harmonic] = eps_harm

    # Write back smoothed values
    eps[ii, jj] = eps_eff

    return np.sqrt(eps)


def _compute_winner_takes_all_n(
    mesh: Mesh2D,
    m_full: IntArray2D,
    materials: dict[Material, int],
    env: Environment,
    structures: list[Structure2D],
    component: ComponentType,
) -> ComplexArray2D:
    """Compute refractive index using winner-takes-all (no subpixel smoothing).

    For each dual cell:
      1. Compute overlap fraction of each material polygon with the cell.
      2. If no single material covers >= 50%, assign background index.
      3. Otherwise, pick the material with the highest overlap.
      4. Ties are broken by mesh order (higher wins).
    """
    si, sj = _COMPONENT_SLICES[component]
    m_comp = m_full[si, sj]

    # Build eps array from material indices (start with non-interface values)
    env_eps = np.complex128(1.0) ** 2
    eps = np.full_like(m_comp, env_eps, dtype=np.complex128)
    mat_eps: dict[int, complex] = {0: env_eps}
    for material, idx in materials.items():
        n_val = material(env, component=component)
        e = n_val**2
        eps[m_comp == idx] = e
        mat_eps[idx] = e

    # Identify interface pixels
    padded = np.pad(m_comp, 1, mode="edge")
    is_interface = (
        (m_comp != padded[:-2, 1:-1])
        | (m_comp != padded[2:, 1:-1])
        | (m_comp != padded[1:-1, :-2])
        | (m_comp != padded[1:-1, 2:])
    )

    if not np.any(is_interface):
        return np.sqrt(eps)

    # Compute dual cell bounds and interface pixel indices
    x_lo, x_hi, y_lo, y_hi = _dual_cell_bounds(mesh, component)
    ii, jj = np.where(is_interface)
    dual_boxes = shapely.box(x_lo[ii], y_lo[jj], x_hi[ii], y_hi[jj])
    dual_areas = shapely.area(dual_boxes)

    # Compute area fractions per material
    effective_polys = _effective_material_polygons(structures, materials)

    # Build mesh-order lookup: mat_idx -> max mesh_order among its structures
    mat_mesh_order: dict[int, int] = {0: -1}  # background has lowest priority
    for s in structures:
        idx = materials[s.material]
        mat_mesh_order[idx] = max(mat_mesh_order.get(idx, -1), s.mesh_order)

    fractions: dict[int, np.ndarray] = {}
    total_struct_fraction = np.zeros(len(ii), dtype=float)
    for idx, poly in effective_polys.items():
        intersections = shapely.intersection(poly, dual_boxes)
        areas = shapely.area(intersections)
        frac = areas / dual_areas
        fractions[idx] = frac
        total_struct_fraction += frac

    total_struct_fraction = np.clip(total_struct_fraction, 0.0, 1.0)
    fractions[0] = np.maximum(1.0 - total_struct_fraction, 0.0)

    # Winner-takes-all: pick material with highest overlap per pixel
    n_pixels = len(ii)
    best_idx = np.zeros(n_pixels, dtype=int)  # 0 = background
    best_frac = fractions[0].copy()
    best_mesh_order = np.full(n_pixels, mat_mesh_order[0])

    for idx, frac in fractions.items():
        if idx == 0:
            continue
        mo = mat_mesh_order.get(idx, -1)
        # Win if: higher fraction, or same fraction but higher mesh order
        wins = (frac > best_frac) | ((frac == best_frac) & (mo > best_mesh_order))
        best_idx[wins] = idx
        best_frac[wins] = frac[wins]
        best_mesh_order[wins] = mo

    # If no material covers >= 50%, fall back to background
    below_threshold = best_frac < 0.5
    best_idx[below_threshold] = 0

    # Assign eps from winner
    winner_eps = np.array([mat_eps[idx] for idx in best_idx])
    eps[ii, jj] = winner_eps

    return np.sqrt(eps)


def _effective_material_polygons(
    structures: list[Structure2D],
    materials: dict[Material, int],
) -> dict[int, shapely.Geometry]:
    """Create non-overlapping effective polygons per material.

    The effective polygons follow the same overwrite precedence as
    `_create_full_material_array`: later groups overwrite earlier groups.
    """
    grouped = _classify_structures_by_mesh_order_and_material(structures, materials)
    ordered_groups: list[tuple[int, shapely.Geometry]] = []
    for group_structs in grouped.values():
        mat_idx = materials[group_structs[0].material]
        poly: Any = unary_union([s.geometry._shapely_polygon() for s in group_structs])
        ordered_groups.append((mat_idx, poly))

    effective_by_mat: dict[int, shapely.Geometry] = {}
    occupied = shapely.GeometryCollection()
    for mat_idx, poly in reversed(ordered_groups):
        eff = poly.difference(occupied)
        if not eff.is_empty:
            if mat_idx in effective_by_mat:
                effective_by_mat[mat_idx] = unary_union(
                    [effective_by_mat[mat_idx], eff]
                )
            else:
                effective_by_mat[mat_idx] = eff
        occupied = unary_union([occupied, poly])
    return effective_by_mat
