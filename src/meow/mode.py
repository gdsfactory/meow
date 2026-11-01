"""An EigenMode."""

from __future__ import annotations

import pickle
import warnings
from collections.abc import Callable
from itertools import product
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.axes import Axes
from pydantic import Field
from scipy.constants import epsilon_0 as eps0
from scipy.constants import mu_0 as mu0

from meow.arrays import Complex, ComplexArray2D, FloatArray2D
from meow.base_model import BaseModel, cached_property
from meow.cross_section import CrossSection
from meow.environment import Environment
from meow.mesh import Mesh2D


def _power(amp: np.ndarray) -> np.ndarray:
    """Calculate the power from the amplitude."""
    return np.abs(amp) ** 2


class Mode(BaseModel):
    """A `Mode` contains the field information for a given `CrossSection`."""

    neff: Complex = Field(description="the effective index of the mode")
    cs: CrossSection = Field(
        description="the index cross section for which the mode was calculated"
    )
    Ex: ComplexArray2D = Field(description="the Ex-fields of the mode")
    Ey: ComplexArray2D = Field(description="the Ey-fields of the mode")
    Ez: ComplexArray2D = Field(description="the Ez-fields of the mode")
    Hx: ComplexArray2D = Field(description="the Hx-fields of the mode")
    Hy: ComplexArray2D = Field(description="the Hy-fields of the mode")
    Hz: ComplexArray2D = Field(description="the Hz-fields of the mode")
    interpolation: Literal["Ex", "Ey", "Ez", "Hz", ""] = Field(
        default="",
        description="To which 2D Yee-location the fields are interpolated to.",
    )

    def interpolate(
        self,
        location: Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"] | None,
    ) -> Mode:
        """Interpolate the mode to the given location."""
        if location is None or location == self.interpolation:
            return self
        if self.interpolation != "":
            msg = "Cannot interpolate from already interpolated mode!"
            raise RuntimeError(msg)
        interpolate_funcs = {
            "EX": _interpolate_Ex,
            "EY": _interpolate_Ey,
            "EZ": _interpolate_Ez,
            "HX": _interpolate_Ey,
            "HY": _interpolate_Ex,
            "HZ": _interpolate_Hz,
        }
        interpolate_func = interpolate_funcs[location.upper()]
        return interpolate_func(self)

    @property
    def te_fraction(self) -> float:
        """The TE polarization fraction of the mode."""
        return te_fraction(self)

    @cached_property
    def _pointing(self) -> tuple[ComplexArray2D, ComplexArray2D, ComplexArray2D]:
        """Calculate and cache the poynting vector."""
        vecE = np.stack([self.Ex, self.Ey, self.Ez], axis=-1)
        vecH = np.stack([self.Hx, self.Hy, self.Hz], axis=-1)
        vecP = np.cross(vecE, vecH)
        Px, Py, Pz = np.rollaxis(vecP, -1)
        return Px, Py, Pz

    @property
    def Px(self) -> ComplexArray2D:
        """The x-component of the Poynting vector."""
        return self._pointing[0]

    @property
    def Py(self) -> ComplexArray2D:
        """The y-component of the Poynting vector."""
        return self._pointing[1]

    @property
    def Pz(self) -> ComplexArray2D:
        """The z-component of the Poynting vector."""
        return self._pointing[2]

    @property
    def env(self) -> Environment:
        """The environment of the mode."""
        return self.cs.env

    @property
    def mesh(self) -> Mesh2D:
        """The mesh of the mode."""
        return self.cs.mesh

    def _visualize(
        self,
        *,
        fields: tuple[str, ...] = ("Ex",),
        ax: Any = None,
        show: bool = True,
        **kwargs: Any,
    ) -> None:
        """Visualization function for Mode, drawing one plot per field.

        Args:
            fields (tuple[str, ...], optional): Tuple of field components to
                visualize. Can be "<E or H or P><x or y or z>. Defaults to ("Ex",).
            ax (Any, optional): Matplotlib axes to draw on: a single Axes, or a
                (nested) list or array of Axes with at least one per field.
                Defaults to None, in which case new axes are made.
            show (bool, optional): Whether to display the plot immediately.
                Defaults to True.
            **kwargs: Visualization options forwarded to `Mode._visualize_field`
                (title, operation, draw_mode_image, ...); see there.
        """
        from meow.visualization import _figsize_visualize_mode

        if not fields:
            fields = ("Ex",)

        if len(fields) == 1:
            if ax is None:
                ax = plt.gca()
            else:
                # ax might be a (nested) list or array of axes; use the first one
                axes = _flatten_axes(ax)
                if not axes:
                    msg = "Empty sequence of axes supplied."
                    raise ValueError(msg)
                ax = axes[0]
            self._visualize_field(fields[0], ax=ax, **kwargs)
        else:
            W, H = _figsize_visualize_mode(self.cs, 6.4)
            if len(fields) > 2:
                max_width = 15
                current_width = len(fields) * W
                W, H = _figsize_visualize_mode(self.cs, 6.4 * max_width / current_width)
            if ax is None:
                _fig, ax = plt.subplots(
                    1,
                    len(fields),
                    figsize=(len(fields) * W, H),
                    constrained_layout=True,
                )
            axes = _flatten_axes(ax)
            if len(axes) < len(fields):
                msg = (
                    f"Not enough axes supplied for the number of fields "
                    f"to plot! {len(axes)} < {len(fields)}."
                )
                raise ValueError(msg)
            for field, ax_ in zip(fields, axes, strict=False):
                self._visualize_field(field, ax=ax_, **kwargs)

        if show:
            plt.show()

    def _visualize_field(
        self,
        field: str,
        *,
        ax: Axes,
        title: str | None = None,
        title_prefix: str = "",
        operation: Callable = _power,
        draw_refractive_index: bool | None = None,
        n_cmap: str | colors.LinearSegmentedColormap | None = None,
        draw_mode_contours: bool = True,
        mode_cmap: str = "inferno",
        num_levels: int = 8,
        draw_mode_image: bool = False,
        mode_image_alpha: float = 0.6,
        draw_structure_boundaries: bool = False,
        structure_boundaries_color: str | tuple[float, ...] = "black",
        **_: Any,
    ) -> None:
        """Draw a single field component of the mode on the given axes.

        Args:
            field (str): Field component to draw: "<E or H or P><x or y or z>".
            ax (Axes): Matplotlib axes to draw on.
            title (str | None, optional): Title for plot image. Defaults to None, in
                which case title is generated using field plotted and effective
                index.
            title_prefix (str, optional): Prefix to put before the automatic title
                (or manual title if specified). Defaults to "".
            operation (Callable, optional): Vectorized function to apply to mode
                field. Defaults to |...|^2.
            draw_refractive_index (bool | None, optional): Whether to draw the
                refractive index profile below the mode (equivalent to visualizing
                the mode's CrossSection underneath the mode). Defaults to None, in
                which case it becomes True iff draw_mode_image is False.
            n_cmap (str | colors.LinearSegmentedColormap | None):
                Maps refractive index to color. Defaults to None.
            draw_mode_contours (bool, optional): Whether to draw mode contours.
                Defaults to True.
            mode_cmap (str, optional): Colormap for the mode, used for both the mode
                image and the mode contours. Defaults to "inferno".
            num_levels (int, optional): Number of contour levels for mode
                contours. Defaults to 8.
            draw_mode_image (bool, optional): Whether to draw the mode as an image.
                Defaults to False.
            mode_image_alpha (float, optional): Transparency of the mode image.
                Defaults to 0.6.
            draw_structure_boundaries (bool, optional): Whether to draw lines around
                structure boundaries. Defaults to False.
            structure_boundaries_color (str | tuple[float, ...], optional):
                Structure boundaries color. Defaults to "black".
        """
        if not (draw_mode_image or draw_mode_contours):
            msg = (
                "Nothing to visualize for the mode: at least one of "
                "draw_mode_image or draw_mode_contours must be True. "
                "To plot only the cross section, visualize mode.cs instead."
            )
            raise ValueError(msg)

        # by default, draw refractive index if not drawing mode
        if draw_refractive_index is None:
            draw_refractive_index = not draw_mode_image

        valid_fields = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "Px", "Py", "Pz"]
        if field not in valid_fields:
            msg = f"Invalid field {field!r}. Valid fields: {', '.join(valid_fields)}."
            raise ValueError(msg)

        plt.sca(ax)

        # first, draw cross section

        if draw_refractive_index:
            if n_cmap is None:
                # little bit lighter colored than the one in cs._visualize:
                n_cmap = colors.LinearSegmentedColormap.from_list(
                    name="c_cmap", colors=["#ffffff", "#c1d9ed"]
                )
            self.cs._visualize(ax=ax, n_cmap=n_cmap, cbar=False, show=False)

        # draw structure boundaries
        if draw_structure_boundaries:
            for structure2d in self.cs.structures:
                boundary = structure2d.geometry._shapely_polygon().exterior
                plt.plot(*boundary.xy, color=structure_boundaries_color, linewidth=1)

        # plot mode image then contours

        x, y = "x", "y"  # currently only propagation in z supported, see Mesh2D
        c = {
            "Ex": "x",
            "Ey": "y",
            "Ez": "z",
            "Hx": "y",
            "Hy": "x",
            "Hz": "z_",
            "Px": "x",
            "Py": "y",
            "Pz": "z",
        }[field]
        X = getattr(self.mesh, f"X{c}")
        Y = getattr(self.mesh, f"Y{c}")
        mode = operation(getattr(self, field))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            levels = np.linspace(mode.min(), mode.max(), num_levels + 1)[1:]

            mode_mappable = None
            if draw_mode_image:
                mode_mappable = plt.pcolormesh(
                    X,
                    Y,
                    mode,
                    cmap=mode_cmap,
                    alpha=mode_image_alpha,
                    # above everything that's already been plotted, below contours
                    zorder=3,
                )

            if draw_mode_contours:
                contours = plt.contour(
                    X,
                    Y,
                    mode,
                    cmap=mode_cmap,
                    levels=levels,
                    zorder=4,  # above mode image
                )
                if mode_mappable is None:
                    mode_mappable = contours

        # colorbar on right, explicitly for the mode (prefer the continuous image;
        # it keeps the image's alpha so it matches what's drawn). Passing ax= lets
        # the figure's layout engine (e.g. constrained layout) reserve its space.

        if mode_mappable is None:  # unreachable: checked at the top of the method
            msg = "No mode image or contours were drawn."
            raise RuntimeError(msg)
        ax.figure.colorbar(mode_mappable, ax=ax, fraction=0.05, pad=0.02)
        plt.sca(ax)

        # labels

        plt.xlabel(x)
        plt.ylabel(y)
        plt.grid(visible=True, alpha=0.4)
        if title is None:
            plt.title(f"{title_prefix}{field} [neff={float(np.real(self.neff)):.6f}]")
        else:
            plt.title(f"{title_prefix}{title}")

        plt.axis("scaled")
        plt.xlim(X.min(), X.max())
        plt.ylim(Y.min(), Y.max())

    def save(self, filename: str | Path) -> None:
        """Save the mode to a file."""
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self))

    @classmethod
    def load(cls, filename: str | Path) -> Mode:
        """Load a mode from a file."""
        return cast(Mode, pickle.loads(Path(filename).read_bytes()))

    def __add__(self, other: Mode) -> Mode:
        if not isinstance(other, Mode):
            msg = (
                "unsupported operand type(s) for +: "
                f"'Mode' and '{type(other).__name__}'"
            )
            raise TypeError(msg)
        new_mode = Mode(
            neff=0.5 * (self.neff + other.neff),
            cs=self.cs,
            Ex=self.Ex + other.Ex,
            Ey=self.Ey + other.Ey,
            Ez=self.Ez + other.Ez,
            Hx=self.Hx + other.Hx,
            Hy=self.Hy + other.Hy,
            Hz=self.Hz + other.Hz,
        )
        return new_mode

    def __mul__(self, other: np.complexfloating | complex) -> Mode:
        if not isinstance(
            other, float | np.complexfloating | np.floating | complex | int | np.integer
        ):
            msg = (
                "unsupported operand type(s) for *: "
                f"'Mode' and '{type(other).__name__}'"
            )
            raise TypeError(msg)
        new_mode = Mode(
            neff=self.neff,
            cs=self.cs,
            Ex=self.Ex * other,
            Ey=self.Ey * other,
            Ez=self.Ez * other,
            Hx=self.Hx * other,
            Hy=self.Hy * other,
            Hz=self.Hz * other,
        )
        return new_mode

    def __rmul__(self, other: np.complexfloating | complex) -> Mode:
        return self.__mul__(other)

    def __truediv__(self, other: np.complexfloating | complex) -> Mode:
        if not isinstance(
            other,
            float | np.complexfloating | np.floating | complex | int | np.integer,
        ):
            msg = (
                "unsupported operand type(s) for /: "
                f"'Mode' and '{type(other).__name__}'"
            )
            raise TypeError(msg)
        return self * (1 / other)

    def __sub__(self, other: Mode) -> Mode:
        if not isinstance(other, Mode):
            msg = (
                "unsupported operand type(s) for -: "
                f"'Mode' and '{type(other).__name__}'"
            )
            raise TypeError(msg)
        return self + other * (-1.0)


Modes: TypeAlias = list[Mode]
"""A list of `Mode` objects."""


def zero_phase(mode: Mode) -> Mode:
    """Normalize (zero out) the phase of a `Mode`.

    The phase is chosen such that the dominant electric field component (the
    largest of Ex, Ey, Ez at the energy density maximum) is real and positive
    there.

    Args:
        mode: the mode to normalize the phase of.

    Returns:
        A new mode with zeroed-out phase.
    """
    e = np.abs(energy_density(mode))
    m, n = np.array(np.where(e == e.max()))[:, 0]
    field_component = max(("Ex", "Ey", "Ez"), key=lambda c: abs(getattr(mode, c)[m, n]))
    phase = np.exp(-1j * np.angle(getattr(mode, field_component)[m, n]))
    new_mode = Mode(
        neff=mode.neff,
        cs=mode.cs,
        Ex=mode.Ex * phase,
        Ey=mode.Ey * phase,
        Ez=mode.Ez * phase,
        Hx=mode.Hx * phase,
        Hy=mode.Hy * phase,
        Hz=mode.Hz * phase,
    )
    if _sum_around(np.real(getattr(new_mode, field_component)), int(m), int(n)) < 0:
        new_mode = invert_mode(new_mode)
    return new_mode


# def _centroid_idxs(arr2d: np.ndarray) -> tuple[int, int]:
#    centroid_x = np.average(np.arange(arr2d.shape[1]), weights=arr2d.sum(axis=0))
#    centroid_y = np.average(np.arange(arr2d.shape[0]), weights=arr2d.sum(axis=1))
#    return round(float(centroid_x)), round(float(centroid_y))


def _sum_around(field: FloatArray2D, m: int, n: int, r: int = 2) -> float:
    total = 0
    idxs = range(-r, r + 1)
    idx_tups = product(idxs, idxs)
    M, N = field.shape
    for i, j in idx_tups:
        m_ = min(m + i, M - 1) if i >= 0 else max(m + i, 0)
        n_ = min(n + j, N - 1) if j >= 0 else max(n + j, 0)
        total = total + field[m_, n_]
    return total


def invert_mode(mode: Mode) -> Mode:
    """Invert a `Mode`.

    Args:
        mode: the mode to invert.

    Returns:
        A new mode with all field components negated.
    """
    return Mode(
        neff=mode.neff,
        cs=mode.cs,
        Ex=-mode.Ex,
        Ey=-mode.Ey,
        Ez=-mode.Ez,
        Hx=-mode.Hx,
        Hy=-mode.Hy,
        Hz=-mode.Hz,
    )


def _crop_non_pml(
    mesh: Mesh2D,
    arr: ComplexArray2D,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[ComplexArray2D, np.ndarray, np.ndarray]:
    numx, numy = mesh.num_pml

    xs = slice(numx, -numx if numx > 0 else None)
    ys = slice(numy, -numy if numy > 0 else None)

    return arr[xs, ys], x[xs], y[ys]


def _canonical_interpolation(
    interpolation: Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"] | None,
) -> Literal["Ex", "Ey", "Ez", "Hz"] | None:
    return cast(
        Literal["Ex", "Ey", "Ez", "Hz"],
        {
            "Ex": "Ex",
            "Ey": "Ey",
            "Ez": "Ez",
            "Hx": "Ey",
            "Hy": "Ex",
            "Hz": "Hz",
        }.get(str(interpolation)),
    )


def inner_product(
    mode1: Mode,
    mode2: Mode,
    *,
    symmetric: bool = False,
    conjugate: bool = False,
    ignore_pml: bool = True,
    interpolation: Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"] | None = None,
) -> complex:
    """The modal inner product for z-normal mode planes.

    Args:
        mode1: the left mode
        mode2: the right mode
        symmetric: use the symmetric inner product definition
        conjugate: use the conjugage inner product definition (power normalization)
        ignore_pml: ignore PML region while taking the inner product
        interpolation: interpolate both modes to a certain field-grid position
            before taking the inner product.

    Returns:
        the inner product
    """
    mode1 = mode1.interpolate(interpolation)
    mode2 = mode2.interpolate(interpolation)

    mesh = mode1.mesh
    x = np.asarray(mesh.x_)
    y = np.asarray(mesh.y_)

    if conjugate:
        ex1, ey1 = mode1.Ex.conj(), mode1.Ey.conj()
        hx1, hy1 = mode1.Hx.conj(), mode1.Hy.conj()
    else:
        ex1, ey1 = mode1.Ex, mode1.Ey
        hx1, hy1 = mode1.Hx, mode1.Hy

    e1_cross_h2 = ex1 * mode2.Hy - ey1 * mode2.Hx
    if symmetric:
        h1_cross_e2 = hx1 * mode2.Ey - hy1 * mode2.Ex
        integrand = 0.25 * (e1_cross_h2 - h1_cross_e2)
    else:
        integrand = 0.5 * e1_cross_h2

    if ignore_pml:
        integrand, x, y = _crop_non_pml(mesh, integrand, x, y)

    arr = np.trapezoid(np.trapezoid(integrand, y), x)
    return float(np.real(arr)) + float(np.imag(arr)) * 1j


def normalize(mode: Mode, inner_product: Callable) -> Mode:
    """Normalize a `Mode` according to the `inner_product` with itself.

    Args:
        mode: the mode to normalize.
        inner_product: a callable computing the inner product of two modes.

    Returns:
        A new mode normalized such that its self-inner-product is unity.
    """
    factor = np.sqrt(inner_product(mode, mode))
    return Mode(
        neff=mode.neff,
        cs=mode.cs,
        Ex=mode.Ex / factor,
        Ey=mode.Ey / factor,
        Ez=mode.Ez / factor,
        Hx=mode.Hx / factor,
        Hy=mode.Hy / factor,
        Hz=mode.Hz / factor,
    )


def electric_energy_density(
    mode: Mode,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """Get the electric energy density contained in a `Mode`.

    Args:
        mode: the mode to compute the electric energy density of.

    Returns:
        A 2D array of electric energy density values.
    """
    epsx, epsy, epsz = mode.cs.nx**2, mode.cs.ny**2, mode.cs.nz**2
    return np.real(
        0.5
        * eps0
        * (
            epsx * np.abs(mode.Ex) ** 2
            + epsy * np.abs(mode.Ey) ** 2
            + epsz * np.abs(mode.Ez) ** 2
        )
    )


def electric_energy(mode: Mode) -> float:
    """Get the electric energy contained in a `Mode`.

    Args:
        mode: the mode to compute the electric energy of.

    Returns:
        The total electric energy summed over the cross section.
    """
    return electric_energy_density(mode).sum()


def magnetic_energy_density(
    mode: Mode,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """Get the magnetic energy density contained in a `Mode`.

    Args:
        mode: the mode to compute the magnetic energy density of.

    Returns:
        A 2D array of magnetic energy density values.
    """
    return np.real(
        0.5 * mu0 * (np.abs(mode.Hx) ** 2 + np.abs(mode.Hy) ** 2 + np.abs(mode.Hz) ** 2)
    )


def magnetic_energy(mode: Mode) -> float:
    """Get the magnetic energy contained in a `Mode`.

    Args:
        mode: the mode to compute the magnetic energy of.

    Returns:
        The total magnetic energy summed over the cross section.
    """
    return magnetic_energy_density(mode).sum()


def energy_density(mode: Mode) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """Get the energy density contained in a `Mode`.

    Args:
        mode: the mode to compute the energy density of.

    Returns:
        A 2D array of total (electric + magnetic) energy density values.
    """
    return electric_energy_density(mode) + magnetic_energy_density(mode)


def energy(mode: Mode) -> float:
    """Get the energy contained in a `Mode`.

    Args:
        mode: the mode to compute the energy of.

    Returns:
        The total energy summed over the cross section.
    """
    return energy_density(mode).sum()


def normalize_energy(mode: Mode) -> Mode:
    """Normalize a mode according to the energy it contains.

    Args:
        mode: the mode to normalize.

    Returns:
        A new mode with fields scaled so that electric and magnetic energies are 0.5.
    """
    e = np.sqrt(2 * electric_energy(mode))
    h = np.sqrt(2 * magnetic_energy(mode))

    return Mode(
        neff=mode.neff,
        cs=mode.cs,
        Ex=mode.Ex / e,
        Ey=mode.Ey / e,
        Ez=mode.Ez / e,
        Hx=mode.Hx / h,
        Hy=mode.Hy / h,
        Hz=mode.Hz / h,
    )


def pml_fraction(mode: Mode) -> float:
    """Fraction of energy density in the PML region.

    Args:
        mode: the mode to evaluate.

    Returns:
        The fraction of total energy density located in the PML region (0.0 to 1.0).
    """
    numx, numy = mode.mesh.num_pml
    if numx == numy == 0:
        return 0.0
    ed = energy_density(mode)
    m, n = ed.shape
    lft = ed[:numx, :]
    rgt = ed[m - numx :, :]
    top = ed[numx : m - numx, :numy]
    btm = ed[numx : m - numx, n - numy :]
    rest = ed[numx : m - numx, numy : n - numy]
    pml_sum = lft.sum() + rgt.sum() + top.sum() + btm.sum()
    total = pml_sum + rest.sum()
    # probably propper integration considering
    # the size of the mesh cells would be better here
    return float(pml_sum / total) if total > 0 else 1.0


def is_pml_mode(mode: Mode, *, threshold: float = 0.1) -> bool:
    """Check whether a mode can be considered a PML mode.

    Args:
        mode: the mode to check.
        threshold: PML energy fraction above which the mode is considered a PML mode.

    Returns:
        True if the PML energy fraction exceeds the threshold.
    """
    threshold = min(max(float(threshold), 0.0), 1.0)
    if threshold > 0.999:
        return False
    return abs(pml_fraction(mode)) > threshold


def is_lossy_mode(mode: Mode, *, threshold: float = 1.0) -> bool:
    """Check whether a mode can be considered lossy.

    Args:
        mode: the mode to check.
        threshold: imaginary neff value above which the mode is considered lossy.

    Returns:
        True if the absolute imaginary part of neff exceeds the threshold.
    """
    return bool(abs(np.imag(mode.neff)) > threshold)


def te_fraction(mode: Mode) -> float:
    """The TE polarization fraction of the `Mode`.

    Args:
        mode: the mode to compute the TE fraction of.

    Returns:
        The TE fraction (0.0 for pure TM, 1.0 for pure TE).
    """
    epsx = np.real(mode.cs.nx**2)
    e = np.sum(0.5 * eps0 * epsx * np.abs(mode.Ex) ** 2)
    h = np.sum(0.5 * mu0 * np.abs(mode.Hx) ** 2)
    return float(e / (e + h))


def _average(
    field: ComplexArray2D,
    direction: Literal["forward", "backward"] = "forward",
    axis: int = 0,
) -> ComplexArray2D:
    if axis == 1:
        return _average(field.T, direction=direction, axis=0).T
    if axis != 0:
        msg = "axis should be 0 or 1"
        raise ValueError(msg)

    average = 0.5 * (field[1:] + field[:-1])
    zero = np.zeros_like(average[:1])

    _direction = str(direction).lower()
    if _direction == "forward":
        return np.concatenate([zero, average], axis=0)

    if _direction != "backward":
        msg = "direction should be 'forward' or backward"
        raise ValueError(msg)

    return np.concatenate([average, zero], axis=0)


def _interpolate_Ex(mode: Mode) -> Mode:
    # TODO: take grid spacing into account
    Ey_at_Ez = _average(mode.Ey, direction="backward", axis=1)
    Ey_at_Ex = _average(Ey_at_Ez, direction="forward", axis=0)
    Ez_at_Ex = _average(mode.Ez, direction="forward", axis=0)
    Hx_at_Hz = _average(mode.Hx, direction="forward", axis=0)
    Hx_at_Ex = _average(Hx_at_Hz, direction="backward", axis=1)
    Hz_at_Ex = _average(mode.Hz, direction="backward", axis=1)
    return Mode(
        neff=mode.neff,
        cs=mode.cs,
        Ex=mode.Ex,
        Ey=Ey_at_Ex,
        Ez=Ez_at_Ex,
        Hx=Hx_at_Ex,
        Hy=mode.Hy,
        Hz=Hz_at_Ex,
        interpolation="Ex",
    )


def _interpolate_Ey(mode: Mode) -> Mode:
    # TODO: take grid spacing into account
    Ex_at_Ez = _average(mode.Ex, direction="backward", axis=0)
    Ex_at_Ey = _average(Ex_at_Ez, direction="forward", axis=1)
    Ez_at_Ey = _average(mode.Ez, direction="forward", axis=1)
    Hy_at_Hz = _average(mode.Hy, direction="forward", axis=1)
    Hy_at_Ey = _average(Hy_at_Hz, direction="backward", axis=0)
    Hz_at_Ey = _average(mode.Hz, direction="backward", axis=0)
    return Mode(
        neff=mode.neff,
        cs=mode.cs,
        Ex=Ex_at_Ey,
        Ey=mode.Ey,
        Ez=Ez_at_Ey,
        Hx=mode.Hx,
        Hy=Hy_at_Ey,
        Hz=Hz_at_Ey,
        interpolation="Ey",
    )


def _interpolate_Ez(mode: Mode) -> Mode:
    # TODO: take grid spacing into account
    return Mode(
        neff=mode.neff,
        cs=mode.cs,
        Ex=_average(mode.Ex, direction="backward", axis=0),
        Ey=_average(mode.Ey, direction="backward", axis=1),
        Ez=mode.Ez,
        Hx=_average(mode.Hx, direction="backward", axis=1),
        Hy=_average(mode.Hy, direction="backward", axis=0),
        Hz=_average(
            _average(mode.Hz, direction="backward", axis=0),
            direction="backward",
            axis=1,
        ),
        interpolation="Ez",
    )


def _interpolate_Hz(mode: Mode) -> Mode:
    # TODO: take grid spacing into account
    return Mode(
        neff=mode.neff,
        cs=mode.cs,
        Ex=_average(mode.Ex, direction="forward", axis=1),
        Ey=_average(mode.Ey, direction="forward", axis=0),
        Ez=_average(
            _average(mode.Ez, direction="forward", axis=0), direction="forward", axis=1
        ),
        Hx=_average(mode.Hx, direction="forward", axis=0),
        Hy=_average(mode.Hy, direction="forward", axis=1),
        Hz=mode.Hz,
        interpolation="Hz",
    )


def _flatten_axes(ax: Any) -> list[Axes]:
    """Flatten an Axes or a (nested) list/array of Axes into a flat list.

    This is used because plt.subplots(...) can return a single Axes or a nested array
    of Axes.
    """
    if isinstance(ax, Axes):
        return [ax]
    flat = list(np.ravel(np.asarray(ax, dtype=object)))
    for a in flat:
        if not isinstance(a, Axes):
            msg = f"Expected matplotlib Axes, got {type(a).__name__}."
            raise TypeError(msg)
    return flat
