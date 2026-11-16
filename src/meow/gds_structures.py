"""GDS Extrusions."""

from __future__ import annotations

import sys
from typing import Any, TypeAlias

import numpy as np
import shapely.geometry as sg
from pydantic import Field
from shapely import make_valid, unary_union

from meow.arrays import FloatArray2D
from meow.base_model import BaseModel
from meow.geometries import Prism
from meow.materials import Material
from meow.structures import Structure3D

# A polygon can reach an extrusion rule either as raw (N, 2) vertices or as an
# already-built shapely polygon. The latter is what `extrude_gdsf_component`
# produces, because that is the only representation that can carry holes.
PolygonInput: TypeAlias = FloatArray2D | sg.Polygon
"""A single polygon: ``(N, 2)`` vertex array, or a shapely polygon (may have holes)."""

PolygonsInput: TypeAlias = PolygonInput | list[PolygonInput]
"""One polygon or a list of them."""

# TODO: Maybe it makes more sense to use native GDSFactory tooling for this


class GdsExtrusionRule(BaseModel):
    """A `GdsExtrusionRule` describes a single extrusion rule.

    Multiple of such rules can later be associated with a gds layer tuple.
    """

    material: Material = Field(description="the material of the extrusion")
    h_min: float = Field(description="the extrusion starting height")
    h_max: float = Field(description="the extrusion ending height")
    buffer: float = Field(
        default=0.0,
        description="an extra buffer (grow / shrink) operation applied to the polygon",
    )
    mesh_order: int = Field(
        default=5, description="the mesh order of the resulting `Structure3D`"
    )

    def __call__(self, polys: PolygonsInput) -> list[Structure3D]:
        """Apply the extrusion rule to one or more polygons.

        Args:
            polys: a single polygon or a list of polygons. Each polygon is
                either an ``(N, 2)`` vertex array or a shapely
                :class:`~shapely.geometry.Polygon`. Shapely polygons are the
                only representation that can carry holes, so they are what
                :func:`extrude_gdsf_component` passes for gdsfactory >= 8.

        Returns:
            One `Structure3D` per connected part of the buffered polygon.
        """
        if isinstance(polys, (np.ndarray, sg.base.BaseGeometry)):
            polys = [polys]

        # make_valid heals holes, turning a polygon that traces a hole and
        # returns back to the outer boundary into an outer boundary polygon
        # with separate hole objects. It is a no-op for already-valid polygons,
        # so passing shapely polygons through it is safe.
        merged = unary_union([make_valid(_as_shapely(poly)) for poly in polys])

        buffered_polygon = merged.buffer(self.buffer)

        # A negative buffer can pinch a shape apart, in which case `buffer()`
        # returns a MultiPolygon. Each part is disconnected from the others and
        # so cannot share a single `Prism`; emit one structure per part rather
        # than silently dropping any of them.
        parts = (
            list(buffered_polygon.geoms)
            if isinstance(buffered_polygon, sg.MultiPolygon)
            else [buffered_polygon]
        )

        return [
            Structure3D(
                material=self.material,
                geometry=Prism(
                    poly=np.asarray(part.exterior.coords),
                    interiors=[
                        np.asarray(interior.coords) for interior in part.interiors
                    ],
                    h_min=self.h_min,
                    h_max=self.h_max,
                    axis="y",
                ),
                mesh_order=self.mesh_order,
            )
            for part in parts
            if part.area > 0
            # ensuring area > 0 removes parts that got shrunken down to 0 area by buffer
        ]


def _as_shapely(poly: PolygonInput) -> sg.Polygon:
    """Normalize a polygon given as raw vertices or as a shapely polygon.

    Args:
        poly: an ``(N, 2)`` vertex array, or an already-built shapely polygon.

    Returns:
        The shapely polygon. Passing an ``sg.Polygon`` through is a no-op, so
        its interior (hole) rings are preserved.

    Raises:
        TypeError: if ``poly`` is neither an array-like nor a shapely polygon.
    """
    if isinstance(poly, sg.Polygon):
        return poly
    if isinstance(poly, sg.base.BaseGeometry):
        # MultiPolygon / GeometryCollection etc. cannot be a single Prism
        msg = (
            f"expected a polygon, got {poly.geom_type}; pass its parts "
            f"separately instead"
        )
        raise TypeError(msg)
    return sg.Polygon(poly)


def _polygon_to_shapely(polygon: Any, scale: float = 1.0) -> sg.Polygon:
    """Convert a gdsfactory/klayout polygon to a shapely polygon, holes included.

    A klayout polygon stores its outline in the *hull* and any cut-outs in
    separate *hole* rings.

    The loop must be bounded by ``polygon.holes()``: ``each_point_hole(index)``
    does not bounds-check, so an out-of-range index reads out of bounds and
    faults with SIGSEGV/SIGBUS from within klayout instead of raising. An
    unguarded loop such as ``itertools.count()`` can therefore kill the
    interpreter.

    Args:
        polygon: a klayout polygon as returned by ``Component.get_polygons``.
        scale: factor applied to every coordinate (e.g. the layout database
            unit, to convert database units to microns).

    Returns:
        The polygon with its interior rings attached.
    """
    shell = [(p.x * scale, p.y * scale) for p in polygon.each_point_hull()]
    holes = [
        [(p.x * scale, p.y * scale) for p in polygon.each_point_hole(index)]
        for index in range(polygon.holes())
    ]
    return sg.Polygon(shell, holes)


def extrude_gds(
    cell: Any,
    extrusions: dict[tuple[int, int], list[GdsExtrusionRule]],
) -> list[Structure3D]:
    """Extrude a GDS cell.

    Accepts a :class:`gdsfactory.Component`, a :class:`gdspy.Cell` or a
    :class:`gdstk.Cell` and forwards to the matching ``extrude_*``
    function.\

    Args:
        cell: a gdsfactory Component, gdspy Cell or gdstk Cell to extrude.
        extrusions: the extrusion rules to use.

    Returns:
        A list of Structure3D objects created from the extruded polygons.

    Raises:
        TypeError: if ``cell`` is not one of the supported cell types.
    """
    # Look the types up in ``sys.modules`` instead of importing them: a
    # cell of a given type can only exist if its library is already
    # imported, so this is safe and avoids costly imports to check type.
    gdsfactory_module = sys.modules.get("gdsfactory")
    if gdsfactory_module is not None and isinstance(cell, gdsfactory_module.Component):
        return extrude_gdsf_component(cell, extrusions)

    gdspy_module = sys.modules.get("gdspy")
    if gdspy_module is not None and isinstance(cell, gdspy_module.Cell):
        return extrude_gdspy_cell(cell, extrusions)

    gdstk_module = sys.modules.get("gdstk")
    if gdstk_module is not None and isinstance(cell, gdstk_module.Cell):
        return extrude_gdstk_cell(cell, extrusions)

    msg = (
        "cell must be a gdsfactory.Component, gdspy.Cell or gdstk.Cell, "
        f"got {type(cell)}"
    )
    raise TypeError(msg)


def extrude_gdsf_component(
    component: gdsfactory.Component,  # noqa: F821  # ty: ignore[unresolved-reference]
    extrusions: dict[tuple[int, int], list[GdsExtrusionRule]],
) -> list[Structure3D]:
    """Extrude a gdsfactory Component given a dictionary of extrusion rules.

    Args:
        component: a gdsfactory Component to extrude.
        extrusions: the extrusion rules to use
            (if not given, the example extrusions will be used).

    Returns:
        A list of Structure3D objects created from the extruded polygons.
    """
    structures = []

    if _get_major_gdsfactory_version() < 8:
        polygons_dict = component.get_polygons(by_spec=True, as_array=True)
    else:
        polygons_dict = component.get_polygons(
            by="tuple",
            layers=extrusions.keys(),
        )
    for layer, polygons in polygons_dict.items():
        if layer not in extrusions:
            continue

        dbu = component.layout().dbu
        polygon_points = []
        for polygon in polygons:
            if _get_major_gdsfactory_version() < 8:
                # pre-8 gdsfactory returns raw arrays, which cannot carry holes
                polygon_points.append(np.asarray(polygon, dtype=float) * dbu)
            else:
                polygon_points.append(_polygon_to_shapely(polygon, dbu))

        for extrusion in extrusions[layer]:
            structures.extend(extrusion(polygon_points))
    return structures


def extrude_gdspy_cell(
    cell: gdspy.Cell,  # ty: ignore[unresolved-reference] # noqa: F821
    extrusions: dict[tuple[int, int], list[GdsExtrusionRule]],
) -> list[Structure3D]:
    """Extrude a gdspy Cell given a dictionary of extrusion rules.

    Args:
        cell: a gdspy Cell to extrude.
        extrusions: the extrusion rules to use.

    Returns:
        A list of Structure3D objects created from the extruded polygons.
    """
    structures = []
    for layer, extrusion_rules in extrusions.items():
        polygons = cell.get_polygons(by_spec=layer)
        for extrusion_rule in extrusion_rules:
            structures.extend(extrusion_rule(list(polygons)))

    return structures


def extrude_gdstk_cell(
    cell: gdstk.Cell,  # ty: ignore[unresolved-reference] # noqa: F821
    extrusions: dict[tuple[int, int], list[GdsExtrusionRule]],
) -> list[Structure3D]:
    """Extrude a gdstk Cell given a dictionary of extrusion rules.

    Args:
        cell: a gdstk Cell to extrude.
        extrusions: the extrusion rules to use.

    Returns:
        A list of Structure3D objects created from the extruded polygons.
    """
    structures = []
    for layer, extrusion_rules in extrusions.items():
        polygons = cell.get_polygons(layer=layer[0], datatype=layer[1])
        for extrusion_rule in extrusion_rules:
            structures.extend(extrusion_rule([np.asarray(p.points) for p in polygons]))

    return structures


def _get_major_gdsfactory_version() -> int:
    from gdsfactory.config import __version__

    major_version = int(__version__.split(".")[0])
    return major_version
