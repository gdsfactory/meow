"""MEOW geometries."""

from __future__ import annotations

import warnings
from secrets import token_hex
from typing import Annotated, Any, Literal, TypeAlias, cast

import matplotlib.pyplot as plt
import numpy as np
import shapely
import shapely.geometry as sg
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Rectangle as MplRect
from pydantic import Field

from meow.arrays import BoolArray2D, DType, FloatArray2D, NDArray, Shape
from meow.base_model import BaseModel

AxisDirection: TypeAlias = Literal["x", "y", "z"]
"""Axis direction: ``"x"``, ``"y"``, or ``"z"``."""


class Geometry2DBase(BaseModel):
    """Base class for 2D geometries."""

    def _mask(self, X: FloatArray2D, Y: FloatArray2D) -> BoolArray2D:
        msg = f"{self.__class__.__name__!r} cannot be masked."
        raise NotImplementedError(msg)

    def _shapely_polygon(self) -> sg.Polygon:
        msg = f"{self.__class__.__name__!r} has no shapely polygon."
        raise NotImplementedError(msg)

    def _visualize(
        self,
        *,
        ax: Any = None,
        show: bool = True,
        color: str | None = None,
        **ignored: Any,
    ) -> None:
        msg = f"{self.__class__.__name__!r} cannot be visualized."
        raise NotImplementedError(msg)


class Rectangle(Geometry2DBase):
    """A Rectangle."""

    x_min: float = Field(description="the minimum x-value of the box")
    x_max: float = Field(description="the maximum x-value of the box")
    y_min: float = Field(description="the minimum y-value of the box")
    y_max: float = Field(description="the maximum y-value of the box")

    def _mask(self, X: FloatArray2D, Y: FloatArray2D) -> BoolArray2D:
        mask = (
            (self.x_min <= X)
            & (X <= self.x_max)
            & (self.y_min <= Y)
            & (Y <= self.y_max)
        )
        return mask

    def _shapely_polygon(self) -> sg.Polygon:
        return sg.box(self.x_min, self.y_min, self.x_max, self.y_max)

    def _visualize(
        self,
        *,
        ax: Any = None,
        show: bool = True,
        color: str | None = None,
        **_: Any,
    ) -> None:
        if ax is None:
            ax = plt.gca()
        if color is None:
            color = "grey"
        width = self.x_max - self.x_min
        height = self.y_max - self.y_min
        mpl_rect = MplRect(
            xy=(self.x_min, self.y_min),
            width=width,
            height=height,
            color=color,
        )
        ax.add_patch(mpl_rect)
        ax.set_xlim(self.x_min - 0.1 * width, self.x_max + 0.1 * width)
        ax.set_ylim(self.y_min - 0.1 * width, self.y_max + 0.1 * width)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.grid(visible=True)
        if show:
            plt.show()


class Polygon2D(Geometry2DBase):
    """A 2D polygon defined by a list of vertices."""

    poly: Annotated[NDArray, Shape(-1, 2), DType("float64")] = Field(
        description="the 2D array (Nx2) with polygon vertices"
    )

    def _mask(self, X: FloatArray2D, Y: FloatArray2D) -> BoolArray2D:
        poly = sg.Polygon(self.poly)

        points = shapely.points(X, Y)
        mask = poly.covers(points)

        return np.asarray(mask)

    def _shapely_polygon(self) -> sg.Polygon:
        return sg.Polygon(self.poly)

    def _visualize(
        self,
        *,
        ax: Any = None,
        show: bool = True,
        color: str | None = None,
        **_: Any,
    ) -> None:
        if ax is None:
            ax = plt.gca()
        if color is None:
            color = "grey"

        patch = MplPolygon(self.poly, closed=True, color=color)

        ax.add_patch(patch)

        min_x, max_x = min(self.poly[:, 0]), max(self.poly[:, 0])
        min_y, max_y = min(self.poly[:, 1]), max(self.poly[:, 1])

        extent_x = max_x - min_x
        extent_y = max_y - min_y

        ax.set_xlim(min_x - 0.1 * extent_x, max_x + 0.1 * extent_x)
        ax.set_ylim(min_y - 0.1 * extent_y, max_y + 0.1 * extent_y)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.grid(visible=True)
        if show:
            plt.show()


Geometry2D: TypeAlias = Rectangle | Polygon2D
"""A 2D geometry: either a `Rectangle` or a `Polygon2D`."""


class Geometry3DBase(BaseModel):
    """Base class for 3D geometries."""

    def _project(self, z: float) -> list[Geometry2DBase]:
        msg = f"{self.__class__.__name__!r} cannot be projected to 2D."

        raise NotImplementedError(msg)

    def _visualize(
        self,
        scale: tuple[float, float, float] | None = None,
        **ignored: Any,  # noqa: ARG002
    ) -> None:
        from trimesh.scene import Scene
        from trimesh.transformations import rotation_matrix

        scene = Scene()
        scene.add_geometry(self._trimesh(scale=scale))
        scene.apply_transform(rotation_matrix(-np.pi / 6, (0, 1, 0)))
        scene.show()

    def _lumadd(
        self,
        sim: Any,
        material_name: str,
        mesh_order: int,
        unit: float = 1e-6,
        xyz: str = "yzx",
    ) -> None:
        msg = f"{self.__class__.__name__!r} cannot be added to Lumerical."
        raise NotImplementedError(msg)

    def _trimesh(
        self, color: str | None = None, scale: tuple[float, float, float] | None = None
    ) -> Any:
        msg = f"{self.__class__.__name__!r} cannot be visualized."
        raise NotImplementedError(msg)


class Box(Geometry3DBase):
    """A Box is a simple rectangular cuboid."""

    x_min: float = Field(description="the minimum x-value of the box")
    x_max: float = Field(description="the maximum x-value of the box")
    y_min: float = Field(description="the minimum y-value of the box")
    y_max: float = Field(description="the maximum y-value of the box")
    z_min: float = Field(description="the minimum z-value of the box")
    z_max: float = Field(description="the maximum z-value of the box")

    def _project(self, z: float) -> list[Geometry2DBase]:
        if z < self.z_min or z > self.z_max:
            return []
        rect = Rectangle(
            x_min=self.x_min,
            x_max=self.x_max,
            y_min=self.y_min,
            y_max=self.y_max,
        )
        return [rect]

    def _lumadd(
        self,
        sim: Any,
        material_name: str,
        mesh_order: int,
        unit: float = 1e-6,
        xyz: str = "yzx",
    ) -> None:
        x, y, z = xyz
        name = token_hex(4)
        kwargs = {
            f"{x}_min": float(self.x_min * unit),
            f"{x}_max": float(self.x_max * unit),
            f"{y}_min": float(self.y_min * unit),
            f"{y}_max": float(self.y_max * unit),
            f"{z}_min": float(self.z_min * unit),
            f"{z}_max": float(self.z_max * unit),
        }
        sim.addrect(
            name=name,
            material=material_name,
            override_mesh_order_from_material_database=True,
            mesh_order=mesh_order,
            **kwargs,
        )

    def _trimesh(
        self, color: str | None = None, scale: tuple[float, float, float] | None = None
    ) -> Any:
        from trimesh import Trimesh  # fmt: skip
        from trimesh.creation import extrude_polygon  # fmt: skip

        sx, sy, sz = scale or (1, 1, 1)
        poly = sg.Polygon(
            [
                (self.x_min * sx, self.y_min * sy),
                (self.x_min * sx, self.y_max * sy),
                (self.x_max * sx, self.y_max * sy),
                (self.x_max * sx, self.y_min * sy),
            ],
        )
        prism = extrude_polygon(poly, self.z_max * sz - self.z_min * sz)
        prism = cast(Trimesh, prism.apply_translation((0, 0, self.z_min * sz)))
        if color is not None:
            prism.visual.face_colors = _to_rgba(color)  # type: ignore[invalid-assignment]  # ty: ignore[invalid-assignment]
        return prism


class Prism(Geometry3DBase):
    """A prism is a 2D Polygon extruded along a axis direction ('x', 'y', 'z')."""

    poly: Annotated[NDArray, Shape(-1, 2), DType("float64")] = Field(
        description="the 2D array (Nx2) with polygon vertices"
    )
    interiors: list[Annotated[NDArray, Shape(-1, 2), DType("float64")]] = Field(
        default_factory=list,
        description="hole rings (Nx2 vertex arrays) cut out of the polygon",
    )
    h_min: float = Field(description="the start height of the extrusion")
    h_max: float = Field(description="the end height of the extrusion")
    axis: AxisDirection = Field(
        default="y",
        description="axis along which the polygon will be extruded ('x', 'y', or 'z').",
    )

    def _prism_polygon(self) -> sg.Polygon:
        """Build the (possibly holed) shapely polygon for this prism."""
        return sg.Polygon(
            shell=self.poly, holes=[np.asarray(h) for h in self.interiors]
        )

    def _project_axis_x(self, z: float) -> list[Geometry2DBase]:
        # x, y, z -> y, z, x
        poly = self._prism_polygon()
        # Column 0 is y and column 1 is z here; scan across the polygon's own
        # y-extent (which can differ from the pre-buffer outline after a
        # positive/negative buffer).
        y_min, y_max = self.poly[:, 0].min(), self.poly[:, 0].max()
        line = sg.LineString([(y_min, z), (y_max, z)])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="shapely")
            intersections = poly.intersection(line)

        if not isinstance(intersections, sg.MultiLineString):
            intersection_array = np.asarray(intersections.coords)
            if not intersection_array.shape[0]:
                return []
            intersections = sg.MultiLineString([cast(sg.LineString, intersections)])

        geoms_2d = []
        for intersection in intersections.geoms:
            intersection = np.asarray(intersection.coords)
            if not intersection.shape[0]:
                continue
            (y_min, _), (y_max, _) = intersection
            y_min, y_max = min(y_min, y_max), max(y_min, y_max)
            x_min, x_max = min(self.h_min, self.h_max), max(self.h_min, self.h_max)
            rect = Rectangle(
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
            )
            geoms_2d.append(rect)
        return geoms_2d

    def _project_axis_y(self, z: float) -> list[Geometry2DBase]:
        # x, y, z -> z, x, y
        poly = self._prism_polygon()
        # x, y, z -> z, x, y: column 0 is z and column 1 is x, so scan across
        # the polygon's own x-extent (which can differ from the pre-buffer
        # outline after a positive/negative buffer).
        x_min, x_max = self.poly[:, 1].min(), self.poly[:, 1].max()
        line = sg.LineString([(z, x_min), (z, x_max)])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="shapely")
            intersections = poly.intersection(line)

        if not isinstance(intersections, sg.MultiLineString):
            intersection_array = np.asarray(intersections.coords)
            if not intersection_array.shape[0]:
                return []
            intersections = sg.MultiLineString([cast(sg.LineString, intersections)])

        geoms_2d = []
        for intersection in intersections.geoms:
            intersection = np.asarray(intersection.coords)
            if not intersection.shape[0]:
                continue
            (_, x_min), (_, x_max) = intersection
            x_min, x_max = min(x_min, x_max), max(x_min, x_max)
            y_min, y_max = min(self.h_min, self.h_max), max(self.h_min, self.h_max)
            rect = Rectangle(
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
            )
            geoms_2d.append(rect)
        return geoms_2d

    def _project_axis_z(self, z: float) -> list[Geometry2DBase]:
        # x, y, z -> x, y, z
        if z < self.h_min or z < self.h_max:
            return []
        return [Polygon2D(poly=self.poly)]

    def _project(self, z: float) -> list[Geometry2DBase]:
        if self.axis == "x":
            return self._project_axis_x(z)
        if self.axis == "y":
            return self._project_axis_y(z)
        return self._project_axis_z(z)

    def _lumadd(
        self,
        sim: Any,
        material_name: str,
        mesh_order: int,
        unit: float = 1e-6,
        xyz: str = "yzx",
    ) -> None:
        name = token_hex(4)

        if xyz not in ("xyz", "yzx", "zxy"):
            msg = (
                "Prism axes should be positively oriented when adding to Lumerical. "
                f"Got: {xyz!r}"
            )
            raise ValueError(msg)

        sim.addpoly(
            name=name,
            material=material_name,
            override_mesh_order_from_material_database=True,
            mesh_order=mesh_order,
            use_relative_coordinates=True,
            vertices=np.asarray(self.poly, float) * float(unit),
            x=0.0,
            y=0.0,
            z_min=float(self.h_min * unit),
            z_max=float(self.h_max * unit),
        )

        x, y, z = xyz
        if self.axis == "x":
            z, x, y = x, y, z
        elif self.axis == "y":
            y, z, x = x, y, z
        xyz = f"{x}{y}{z}"

        if xyz in ["yzx", "zxy"]:
            msg = (
                "Only prisms extruded perpendicular to the 'chip surface' "
                "are currently supported in Lumerical."
            )
            raise NotImplementedError(msg)

    def _trimesh(
        self, color: str | None = None, scale: tuple[float, float, float] | None = None
    ) -> Any:
        from trimesh.creation import extrude_polygon  # fmt: skip

        poly = sg.Polygon(self.poly)
        prism = extrude_polygon(poly, self.h_max - self.h_min)
        prism = prism.apply_translation((0, 0, self.h_min))
        if self.axis == "x":
            prism.vertices = np.roll(prism.vertices, shift=1, axis=1)
        if self.axis == "y":
            prism.vertices = np.roll(prism.vertices, shift=-1, axis=1)
        if scale is not None:
            sx, sy, sz = scale
            prism.vertices *= np.array([[sx, sy, sz]])
        if color is not None:
            prism.visual.face_colors = _to_rgba(color)  # type: ignore[invalid-assignment]
        return prism

    def _center(self) -> tuple[float, float, float]:
        import shapely.geometry as sg  # fmt: skip

        a, b = np.array(sg.Polygon(self.poly).centroid.xy).ravel()
        c = 0.5 * (self.h_min + self.h_max)
        x, y, z = {
            "x": (c, a, b),
            "y": (b, c, a),
            "z": (a, b, c),
        }[self.axis]
        return x, y, z

    def _axis_tuple(self) -> tuple[int, int, int]:
        return {
            "x": (1, 0, 0),
            "y": (0, 1, 0),
            "z": (0, 0, 1),
        }[self.axis]


Geometry3D: TypeAlias = Box | Prism
"""A 3D geometry: either a `Box` or a `Prism`."""

Geometry: TypeAlias = Geometry2D | Geometry3D
"""Any 2D or 3D geometry."""


def _to_rgba(c: Any) -> tuple[float, float, float, float]:
    from matplotlib.colors import to_rgba as _to_rgba_mpl  # fmt: skip

    r, g, b, a = _to_rgba_mpl(c)
    a = min(max(a, 0.1), 0.9)
    return float(r), float(g), float(b), float(a)
