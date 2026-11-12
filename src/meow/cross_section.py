"""A CrossSection."""

from __future__ import annotations

from typing import Any, Self

import numpy as np
import shapely
from matplotlib.ticker import FormatStrFormatter, Locator, MaxNLocator
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
from meow.materials import ComponentType, Material, SpatiallyVaryingMaterial
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
    z: float | None = Field(
        default=None,
        description="""The z-position at which this CrossSection is sampled.

        Only used by spatially varying materials that vary along z; it is passed
        as the `z` argument of their index callable. It may be None if the
        z-position is unknown (for example when a CrossSection is constructed
        directly rather than through `from_cell`), in which case a material that
        genuinely depends on z is expected to raise. Materials that do not vary
        along z ignore it.""",
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
            z=cell.z,
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
        """Return the refractive index (along the given axis) for the full mesh.

        This does not do any subpixel smoothing; it returns the raw material values.
        """
        # check if n_full for this component exists in cache
        if component in self._n_full_cache:
            return self._n_full_cache[component]
        n_full = np.full_like(self.mesh.X_full, _VACUUM_EPS, dtype=np.complex128)
        for material, idx in self.materials.items():
            mask = self._m_full == idx
            if not np.any(mask):
                continue
            n_full[mask] = _material_n(
                material,
                self.env,
                component,
                self.mesh.X_full[mask],
                self.mesh.Y_full[mask],
                self.z,
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
                self.mesh,
                self._m_full,
                self.materials,
                self.env,
                self.structures,
                "x",
                z=self.z,
            )
        return _compute_winner_takes_all_n(
            self.mesh,
            self._m_full,
            self.materials,
            self.env,
            self.structures,
            "x",
            z=self.z,
        )

    @cached_property
    def ny(self) -> ComplexArray2D:
        """Return smoothed refractive index on the Ey positions.

        Associated with y axis (for anisotropic materials).
        """
        if self.subpixel_smoothing:
            return _compute_smoothed_n(
                self.mesh,
                self._m_full,
                self.materials,
                self.env,
                self.structures,
                "y",
                z=self.z,
            )
        return _compute_winner_takes_all_n(
            self.mesh,
            self._m_full,
            self.materials,
            self.env,
            self.structures,
            "y",
            z=self.z,
        )

    @cached_property
    def nz(self) -> ComplexArray2D:
        """Return smoothed refractive index on the Ez positions.

        Associated with z axis (for anisotropic materials).
        """
        if self.subpixel_smoothing:
            return _compute_smoothed_n(
                self.mesh,
                self._m_full,
                self.materials,
                self.env,
                self.structures,
                "z",
                z=self.z,
            )
        return _compute_winner_takes_all_n(
            self.mesh,
            self._m_full,
            self.materials,
            self.env,
            self.structures,
            "z",
            z=self.z,
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
            _reserve_colorbar_label_space(ax)
            _cbar = plt.colorbar(cax=cax)
            # Ticks are chosen at draw time, once the bar's height is known, so
            # the bar always fits the labels. With only constant materials,
            # tick each material's index; a spatially varying material yields a
            # continuum of indices, so use round, evenly spaced values instead.
            if any(_is_spatially_varying(m) for m in self.materials):
                _cbar.locator = MaxNLocator(nbins="auto")
            else:
                _cbar.locator = _MaterialIndexLocator(np.unique(n_full))
                _cbar.formatter = FormatStrFormatter("%.3f")
            _cbar.update_ticks()
            plt.sca(ax)
        plt.axis("scaled")
        if show:
            plt.show()


# --- Subpixel permittivity smoothing ---

_VACUUM_EPS = np.complex128(1.0) ** 2
"""Permittivity of the background (vacuum, n=1)."""

_CBAR_LABEL_MARGIN = 0.93
"""Fraction of the layout width to use, leaving room for colorbar labels."""


def _reserve_colorbar_label_space(ax: Any) -> None:
    """Leave room at the figure's right edge for a colorbar's tick labels.

    ``make_axes_locatable`` places its axes immediately right of the parent and
    takes no account of the tick labels that then hang off *that*. Under
    ``constrained_layout`` the parent is sized to fill the figure, so the bar
    ends up flush with the right edge and its labels are drawn past it and
    clipped. Narrowing the layout rectangle reserves the margin.
    """
    engine = ax.figure.get_layout_engine()
    if engine is None:
        return
    options = engine.get()
    if not options or "rect" not in options:
        return
    left, bottom, width, height = options["rect"]
    engine.set(rect=(left, bottom, width * _CBAR_LABEL_MARGIN, height))


class _MaterialIndexLocator(Locator):
    """Colorbar locator that ticks each material's index, as many as fit.

    Matplotlib calls the locator at draw time, after layout, so the number of
    ticks that fit (``axis.get_tick_space()``) reflects the bar's actual
    height. The lowest and highest indices are always kept; the others are
    kept, lowest first, if they are at least one label's height from every
    tick kept so far. That drops indices too close to label separately (e.g.
    1.444 and 1.45) and thins the ticks on a short bar.
    """

    def __init__(self, values: np.ndarray) -> None:
        self.values = np.sort(np.asarray(values, dtype=float))

    def __call__(self) -> list[float]:
        """Return the tick positions for the current view."""
        axis = self.axis
        if axis is None:  # not attached to an axis yet
            return self.values.tolist()
        lo, hi = sorted(axis.get_view_interval())
        values = self.values[(self.values >= lo) & (self.values <= hi)]
        if values.size <= 1:
            return values.tolist()
        min_gap = (hi - lo) / max(axis.get_tick_space(), 1)
        lowest, highest = values[0], values[-1]
        if highest - lowest < min_gap:
            return [lowest]
        kept = [lowest]
        for v in values[1:-1]:
            if v - kept[-1] >= min_gap and highest - v >= min_gap:
                kept.append(v)
        kept.append(highest)
        return kept


def _staggered_coords(
    mesh: Mesh2D, component: ComponentType
) -> tuple[ComplexArray2D, ComplexArray2D]:
    """Return the (X, Y) coordinates of the given field component's positions."""
    coords = {
        "x": (mesh.Xx, mesh.Yx),
        "y": (mesh.Xy, mesh.Yy),
        "z": (mesh.Xz, mesh.Yz),
    }[component]
    return coords


def _is_spatially_varying(material: Material) -> bool:
    """Whether the material defines its index as a coordinate function."""
    return isinstance(material, SpatiallyVaryingMaterial)


def _material_n(
    material: Material,
    env: Environment,
    component: ComponentType,
    x: np.ndarray,
    y: np.ndarray,
    z: float | None,
) -> float | complex | np.ndarray:
    """Refractive index of ``material`` at the given coordinates.

    For a spatially varying material the user callable is evaluated at the given
    coordinates; otherwise the material's scalar index is returned.

    Args:
        material: the material to evaluate.
        env: the environment.
        component: the field component ('x', 'y' or 'z').
        x: the x coordinates to evaluate at.
        y: the y coordinates to evaluate at.
        z: the z-position, or None if unknown.

    Returns:
        A scalar for a constant material, or an array broadcastable to ``x`` for
        a spatially varying material.
    """
    if not _is_spatially_varying(material):
        return material(env, component=component)
    return material.n_at(env, x, y, z, component=component)  # ty: ignore[unresolved-attribute]


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
    z: float | None = None,
) -> ComplexArray2D:
    """Compute subpixel-smoothed refractive index for a field component.

    At interface pixels, the effective permittivity is computed using
    area-fraction-weighted averaging:
      - E parallel to interface:      arithmetic avg  eps_eff = sum(f_i * eps_i)
      - E perpendicular to interface: harmonic avg    1/eps_eff = sum(f_i / eps_i)

    The per-material permittivity is looked up per *pixel* rather than per
    material index, so that a spatially varying material is smoothed against its
    neighbours using its locally sampled permittivity. For materials with a
    constant index the per-pixel value is just that constant, so the result is
    unchanged.
    """
    si, sj = _COMPONENT_SLICES[component]
    m_comp = m_full[si, sj]

    eps, _ = _build_eps(mesh, m_comp, materials, env, component, z)

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

    # Permittivity of each material *at each interface pixel*, so that a
    # spatially varying material contributes its locally sampled value where it
    # meets a neighbour. For a constant material this is just its scalar eps.
    # Note this cannot be read off `eps`, which only holds the rasterizer's
    # winner for the pixel and so would give the wrong material's value.
    eps_at: dict[int, np.ndarray] = {
        idx: np.full(len(ii), complex(_VACUUM_EPS), dtype=np.complex128)
        for idx in fractions
    }
    for material, idx in materials.items():
        if idx not in eps_at:
            continue
        if _is_spatially_varying(material):
            xs_c, ys_c = _staggered_coords(mesh, component)
            # Broadcast: a user callable may legitimately return a scalar even
            # when given coordinate arrays, so the result is expanded to one
            # value per interface pixel rather than left as a 0-d array.
            eps_at[idx] = np.broadcast_to(
                np.asarray(
                    _material_n(material, env, component, xs_c[ii, jj], ys_c[ii, jj], z)
                )
                ** 2,
                (len(ii),),
            )
        else:
            eps_at[idx] = np.full(
                len(ii), complex(np.asarray(material(env, component=component)) ** 2)
            )
    # Background vacuously holds vacuum eps (already initialised above).

    # Compute effective eps for each interface pixel
    eps_eff = np.zeros(len(ii), dtype=np.complex128)

    # Arithmetic average: eps_eff = sum(f_i * eps_i)
    arith_mask = ~use_harmonic
    if np.any(arith_mask):
        eps_arith = np.zeros(arith_mask.sum(), dtype=np.complex128)
        for idx, frac in fractions.items():
            eps_arith += frac[arith_mask] * eps_at[idx][arith_mask]
        eps_eff[arith_mask] = eps_arith

    # Harmonic average: 1/eps_eff = sum(f_i / eps_i)
    if np.any(use_harmonic):
        inv_eps_harm = np.zeros(use_harmonic.sum(), dtype=np.complex128)
        for idx, frac in fractions.items():
            e = eps_at[idx][use_harmonic]
            nonzero = np.abs(e) > 0
            contrib = np.zeros_like(e)
            contrib[nonzero] = 1.0 / e[nonzero]
            inv_eps_harm += frac[use_harmonic] * contrib
        eps_harm = np.zeros_like(inv_eps_harm)
        safe = np.abs(inv_eps_harm) > 1e-30
        eps_harm[safe] = 1.0 / inv_eps_harm[safe]
        eps_harm[~safe] = eps[ii[use_harmonic], jj[use_harmonic]][~safe]
        eps_eff[use_harmonic] = eps_harm

    # Write back smoothed values
    eps[ii, jj] = eps_eff

    return np.sqrt(eps)


def _build_eps(
    mesh: Mesh2D,
    m_comp: IntArray2D,
    materials: dict[Material, int],
    env: Environment,
    component: ComponentType,
    z: float | None,
) -> tuple[ComplexArray2D, dict[int, complex]]:
    """Build the per-pixel permittivity array for a component.

    Returns the eps array and the per-material-index scalar eps (used for
    materials with a constant index; the map is only valid for those).
    """
    eps = np.full_like(m_comp, _VACUUM_EPS, dtype=np.complex128)
    mat_eps: dict[int, complex] = {0: complex(_VACUUM_EPS)}
    xs, ys = _staggered_coords(mesh, component)
    for material, idx in materials.items():
        mask = m_comp == idx
        if not np.any(mask):
            continue
        n_val = _material_n(material, env, component, xs[mask], ys[mask], z)
        e = np.asarray(n_val) ** 2
        eps[mask] = e
        if not _is_spatially_varying(material):
            mat_eps[idx] = complex(e)
    return eps, mat_eps


def _compute_winner_takes_all_n(
    mesh: Mesh2D,
    m_full: IntArray2D,
    materials: dict[Material, int],
    env: Environment,
    structures: list[Structure2D],
    component: ComponentType,
    z: float | None = None,
) -> ComplexArray2D:
    """Compute refractive index using winner-takes-all (no subpixel smoothing).

    For each dual cell:
      1. Compute overlap fraction of each material polygon with the cell.
      2. If no single material covers >= 50%, assign background index.
      3. Otherwise, pick the material with the highest overlap.
      4. Ties are broken by mesh order, highest winning -- note this is the
         OPPOSITE of `_create_full_material_array`, which lets the lowest
         mesh_order win (it applies last-wins over a descending sort). The two
         disagree on ties; see mwe_mesh_order.py. Not yet resolved because
         changing it would alter public behaviour.

    A spatially varying material keeps its per-pixel sampled index where it wins;
    the winner is still chosen per material index, which is well defined because
    a spatially varying material is a single material region.
    """
    si, sj = _COMPONENT_SLICES[component]
    m_comp = m_full[si, sj]

    # Identify interface pixels
    padded = np.pad(m_comp, 1, mode="edge")
    is_interface = (
        (m_comp != padded[:-2, 1:-1])
        | (m_comp != padded[2:, 1:-1])
        | (m_comp != padded[1:-1, :-2])
        | (m_comp != padded[1:-1, 2:])
    )

    if not np.any(is_interface):
        eps, _ = _build_eps(mesh, m_comp, materials, env, component, z)
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

    # Relabel each interface pixel with its winner, then build eps from the
    # labels. The rasterizer's label and the winner can differ (in either
    # direction, background included), so the label must be overwritten.
    m_comp = m_comp.copy()
    m_comp[ii, jj] = best_idx
    eps, _ = _build_eps(mesh, m_comp, materials, env, component, z)
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
