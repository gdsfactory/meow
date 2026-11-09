"""Visualizations for common meow-datatypes."""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Callable, Iterable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import sax
from trimesh.scene import Scene
from trimesh.transformations import rotation_matrix

from meow.arrays import ComplexArray2D
from meow.base_model import BaseModel
from meow.cross_section import CrossSection
from meow.mode import Mode  # fmt: skip
from meow.structures import Structure3D, sort_structures

gf = None
with contextlib.suppress(ImportError):
    import gdsfactory as gf


def _visualize_s_matrix(
    S: ComplexArray2D,
    *,
    fmt: str | None = None,
    title: str | None = None,
    show: bool = True,
    phase: bool = False,
    ax: Any = None,
) -> None:
    fmt = ".0f" if phase else ".3f"

    Z = np.abs(S)
    y, x = np.arange(Z.shape[0])[::-1], np.arange(Z.shape[1])
    Y, X = np.meshgrid(y, x)

    if ax:
        plt.sca(ax)
    else:
        plt.figure(figsize=(2 * x.shape[0] / 3, 2 * y.shape[0] / 3))

    plt.pcolormesh(X, Y, Z[::-1].T, cmap="Greys", vmin=0.0, vmax=2.0 * Z.max())

    coords_ = np.concatenate(
        [np.array([x[0] - 1]), x, np.array([x[-1] + 1])], axis=0, dtype=float
    )
    labels = ["" for _ in coords_]
    plt.xticks(coords_ + 0.5, labels)
    plt.xlim(coords_[0] + 0.5, coords_[-1] - 0.5)

    coords_ = np.concatenate(
        [np.array([y[0] + 1]), y, np.array([y[-1] - 1])], axis=0, dtype=float
    )
    coords_ = coords_[::-1]  # reverse
    labels = ["" for _ in coords_]
    plt.yticks(coords_ + 0.5, labels)
    plt.ylim(coords_[-1] - 0.5, coords_[0] + 0.5)
    plt.grid(visible=True)

    for x, y, z in zip(X.ravel(), Y.ravel(), S[::-1].T.ravel(), strict=False):
        if np.abs(z) > 0.0005:
            if phase:
                z = np.angle(z) * 180 / np.pi
            text = f"{z:{fmt}}"
            text = text.replace("+", "\n+")
            text = text.replace("-", "\n-")
            if text[0] == "\n":
                text = text[1:]
            plt.text(x, y, text, ha="center", va="center", fontsize=8)

    if title is not None:
        plt.title(title)

    if show:
        plt.show()


def _visualize_s_pm_matrix(
    Spm: sax.SDense,
    *,
    fmt: str | None = None,
    title: str | None = None,
    show: bool = True,
    phase: bool = False,
    ax: Any = None,
) -> None:
    S, pm = Spm
    _visualize_s_matrix(
        np.asarray(S), fmt=fmt, title=title, show=False, phase=phase, ax=ax
    )
    num_left = len([p for p in pm if "left" in p])
    Z = np.abs(S)
    _, x = np.arange(Z.shape[0])[::-1], np.arange(Z.shape[1])

    plt.axvline(x[num_left] - 0.5, color="red")
    plt.axhline(x[num_left] - 0.5, color="red")

    if show:
        plt.show()


def _visualize_overlap_density(
    two_modes: tuple[Mode, Mode],
    *,
    conjugated: bool = True,
    x_symmetry: bool = False,
    y_symmetry: bool = False,
    ax: Any = None,
    n_cmap: str | None = None,
    mode_cmap: str = "inferno",
    num_levels: int = 8,
    show: bool = True,
) -> None:
    mode1, mode2 = two_modes
    if conjugated:
        cross = mode1.Ex * mode2.Hy.conj() - mode1.Ey * mode2.Hx.conj()
    else:
        cross = mode1.Ex * mode2.Hy - mode1.Ey * mode2.Hx
    if x_symmetry:
        cross = 0.5 * (cross + cross[::-1])
    if y_symmetry:
        cross = 0.5 * (cross + cross[:, ::-1])
    zeros = np.zeros_like(cross)
    overlap = Mode(
        neff=mode1.neff,
        cs=mode1.cs,
        Ex=cross,
        Ey=zeros,
        Ez=zeros,
        Hx=zeros,
        Hy=zeros,
        Hz=zeros,
    )
    if ax is None:
        W, H = _figsize_visualize_mode(mode1.cs, 5)
        _, ax = plt.subplots(1, 3, figsize=(3 * W, H))

    field = "Ex" if mode1.te_fraction > 0.5 else "Hx"
    mode1._visualize(
        title=f"mode 1: {field}",
        fields=(field,),
        ax=ax[0],
        n_cmap=n_cmap,
        mode_cmap=mode_cmap,
        num_levels=num_levels,
        show=False,
    )

    field = "Ex" if mode2.te_fraction > 0.5 else "Hx"
    mode2._visualize(
        title=f"mode 2: {field}",
        fields=(field,),
        ax=ax[1],
        n_cmap=n_cmap,
        mode_cmap=mode_cmap,
        num_levels=num_levels,
        show=False,
    )

    title = "overlap density" + ("" if conjugated else " (no conjugations)")
    p = overlap._visualize(
        title=title,
        fields=("Ex",),
        ax=ax[2],
        n_cmap=n_cmap,
        mode_cmap=mode_cmap,
        num_levels=num_levels,
        show=False,
    )

    if show:
        plt.show()

    return p


def _visualize_gf_component(comp: Any) -> None:
    if gf is not None:
        comp.plot()


def _is_two_tuple(obj: Any) -> bool:
    if not isinstance(obj, tuple):
        return False
    return len(obj) == 2


def add_right_colorbar(
    ax: Any,
    mappable: Any,
    *,
    ticks: Any = None,
) -> Any:
    """Add a colorbar hugging the right edge of ``ax``.

    Uses an ``inset_axes`` anchored in the parent's own coordinates rather than
    ``figure.colorbar(ax=ax)``. The latter asks the layout engine to reserve
    space, which behaves well in a tight figure but leaves a large gap once the
    figure is wider than the grid needs -- the gap grows with the figure width,
    so resizing the window visibly detaches the bar from its plot. Anchoring to
    ``ax.transAxes`` keeps the bar a fixed small distance from the plot at any
    figure size, which is what makes interactive resizing behave.

    Args:
        ax: the axes the colorbar belongs to.
        mappable: the mappable (e.g. the ``QuadMesh`` from ``pcolormesh``).
        ticks: optional explicit tick locations.

    Returns:
        The created colorbar.
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # fmt: skip

    cax = inset_axes(
        ax,
        width="5%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(1.05, 0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    return ax.figure.colorbar(mappable, cax=cax, ticks=ticks)


def _figsize_visualize_mode(cs: CrossSection, W0: float) -> tuple[float, float]:
    x_min, x_max = cs.mesh.x.min(), cs.mesh.x.max()
    y_min, y_max = cs.mesh.y.min(), cs.mesh.y.max()
    delta_x = x_max - x_min
    delta_y = y_max - y_min
    aspect = delta_y / delta_x
    # extra width for y labels + colorbar, extra height for title + x labels
    W, H = W0 + 1.4, W0 * aspect + 0.7
    return W, H


def _power(amp: np.ndarray) -> np.ndarray:
    return abs(amp) ** 2


def _visualize_modes(
    modes: list[Mode],
    *,
    title_prefix: str = "",
    fields: tuple[str, ...] = ("Ex", "Hx"),
    ax: Any = None,
    show: bool = True,
    plot_width: float = 6.4,
    **kwargs: Any,
) -> None:
    """Visualize a list of modes, one row per mode and one column per field.

    Any other keyword arguments (e.g. title, operation, draw_mode_image) are
    forwarded to `Mode._visualize`; see there for the available options.
    """
    num_modes = len(modes)
    cs = modes[0].cs
    W, H = _figsize_visualize_mode(cs, plot_width)

    if ax is None:
        _fig, ax = plt.subplots(
            num_modes,
            len(fields),
            figsize=(len(fields) * W, num_modes * H),
            sharex=True,
            sharey=True,
            squeeze=False,
            constrained_layout=True,
        )
    else:
        _fig = None
    for i, m in enumerate(modes):
        m._visualize(
            **kwargs,
            title_prefix=title_prefix or f"m{i}: ",
            fields=tuple(fields),
            ax=ax[i],
            show=False,
        )
    if show:
        plt.show()


def _visualize_base_model(obj: BaseModel, **kwargs: Any) -> None:
    return obj._visualize(**kwargs)


def _is_mode_list(obj: Any) -> bool:
    return isinstance(obj, Iterable) and all(isinstance(o, Mode) for o in obj)


def _is_structure_3d_list(obj: Any) -> bool:
    return isinstance(obj, Iterable) and all(isinstance(o, Structure3D) for o in obj)


def _is_base_model(obj: Any) -> bool:
    return isinstance(obj, BaseModel)


def _is_mode_overlap(obj: Any) -> bool:
    return _is_two_tuple(obj) and all(isinstance(o, Mode) for o in obj)


def _is_s_matrix(obj: Any) -> bool:
    arr = np.asarray(obj)
    return arr.ndim == 2 and arr.shape[0] > 1 and arr.shape[1] > 1


def _is_s_pm_matrix(obj: Any) -> bool:
    return _is_two_tuple(obj) and _is_s_matrix(obj[0]) and isinstance(obj[1], dict)


def _is_gf_component(obj: Any) -> bool:
    return gf is not None and isinstance(obj, gf.Component)


def _visualize_structures(
    structures: list[Structure3D],
    scale: tuple[float, float, float] | None = None,
) -> Any:
    """Easily visualize a collection (list) of `Structure3D` objects."""
    scene = Scene(
        geometry=[s._trimesh(scale=scale) for s in sort_structures(structures)]
    )
    scene.apply_transform(rotation_matrix(np.pi - np.pi / 6, (0, 1, 0)))
    return scene.show()


def _get_vis_func(obj: Any) -> Callable:
    """Get the visualization function for the given object."""
    for check_func, vis_func in VISUALIZATION_MAPPING.items():
        if check_func(obj):
            return vis_func
    return _print_obj


def _print_obj(obj: Any) -> None:
    """Print the object representation."""
    sys.stdout.write(f"{obj}\n")


VISUALIZATION_MAPPING: dict[Callable, Callable] = {
    _is_base_model: _visualize_base_model,
    _is_mode_list: _visualize_modes,
    _is_structure_3d_list: _visualize_structures,
    _is_mode_overlap: _visualize_overlap_density,
    _is_s_pm_matrix: _visualize_s_pm_matrix,
    _is_s_matrix: _visualize_s_matrix,
    _is_gf_component: _visualize_gf_component,
}


def visualize(obj: Any, **kwargs: Any) -> Any:
    """Visualize any meow object.

    Args:
        obj: the meow object to visualize
        **kwargs: extra configuration to visualize the object

    Returns:
        The result of the visualization function for the given object.

    Note:
        Most meow objects have a `._visualize` method.
        Check out its help to see which kwargs are accepted.
    """
    try:
        is_empty = bool(not obj)
    except ValueError:
        is_empty = isinstance(obj, np.ndarray) and obj.size == 0

    if is_empty:
        msg = "Nothing to visualize!"
        raise ValueError(msg)

    vis_func = _get_vis_func(obj)
    vis_func(obj, **kwargs)


vis = visualize  # shorthand for visualize
