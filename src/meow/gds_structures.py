"""GDS Extrusions."""

from __future__ import annotations

import sys
from typing import Any

import gdsfactory
import numpy as np
import shapely.geometry as sg
from pydantic import Field
from shapely import make_valid

from meow.arrays import FloatArray2D
from meow.base_model import BaseModel
from meow.geometries import Prism
from meow.materials import Material
from meow.structures import Structure3D

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

    def __call__(self, poly: FloatArray2D) -> Structure3D:
        """Apply the extrusion rule to a polygon."""
        # make_valid heals holes by turning one polygon that traces holes and returns
        # back to outer boudary, into outer boundary polygon with separate hole objects
        shapely_polygon = make_valid(sg.Polygon(poly))

        buffered_polygon = shapely_polygon.buffer(self.buffer)

        # NOTE: `get_coordinates` flattens the exterior ring *and* every
        # interior (hole) ring into one vertex array. Rebuilding a Prism from
        # that concatenation produces a self-touching, invalid polygon (the
        # hole ring is joined back onto the exterior ring), which makes the
        # scans in `Prism._project_axis_*` report spurious extra pieces.
        # Keep the rings separate instead, so the holes stay holes.
        #
        # `buffer()` can return a MultiPolygon (e.g. a thin buffer splitting a
        # shape in two); only the largest part is representable by a single
        # `Prism`, so keep that one.
        if isinstance(buffered_polygon, sg.MultiPolygon):
            buffered_polygon = max(buffered_polygon.geoms, key=lambda g: g.area)

        return Structure3D(
            material=self.material,
            geometry=Prism(
                poly=np.asarray(buffered_polygon.exterior.coords),
                interiors=[
                    np.asarray(interior.coords)
                    for interior in buffered_polygon.interiors
                ],
                h_min=self.h_min,
                h_max=self.h_max,
                axis="y",
            ),
            mesh_order=self.mesh_order,
        )


# Frequenct linter and type checker ignores in following methods are becaues
# we are guarding against older versions of gdsfactory that have different
# method signatures.


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
    component: gdsfactory.Component,
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
        polygons_dict = component.get_polygons(by_spec=True, as_array=True)  # ty: ignore[unknown-argument]
    else:
        polygons_dict = component.get_polygons(
            by="tuple",
            layers=extrusions.keys(),  # ty: ignore[invalid-argument-type]
        )
    for layer, polygons in polygons_dict.items():
        for polygon in polygons:
            if layer not in extrusions:
                continue

            if _get_major_gdsfactory_version() < 8:
                polygon_points = polygon
            else:
                polygon_points = np.array(
                    [[p.x, p.y] for p in polygon.each_point_hull()], dtype=float
                )

            polygon_points *= component.layout().dbu
            structures.extend(
                extrusion(polygon_points)  # ty: ignore[invalid-argument-type]
                for extrusion in extrusions[layer]  # ty: ignore[invalid-argument-type]
            )
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
            structures.extend(extrusion_rule(polygon) for polygon in polygons)

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
            structures.extend(extrusion_rule(polygon.points) for polygon in polygons)

    return structures


def _get_major_gdsfactory_version() -> int:
    from gdsfactory.config import __version__

    major_version = int(__version__.split(".")[0])
    return major_version
