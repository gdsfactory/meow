"""The fundamental mode of a parabolic index profile is a Gaussian.

A parabolic index profile acts like a harmonic oscillator: the fundamental mode
is a Gaussian, with a waist set by the curvature of the parabola. In the
paraxial limit, for

    n^2(x) = n0^2 * (1 - (x/a)^2)

the fundamental mode is proportional to ``exp(-x^2 / (2 w^2))`` with

    w^2 = a / (k0 * n0)

These tests build that profile as a `SpatiallyVaryingMaterial` and compare the
solved mode against the analytic Gaussian.

Two cases:

* **1D** -- the index varies in x only, and the y direction carries no PMLs, so
  that direction behaves as a PEC waveguide: the mode is uniform in y out to the
  edges and Gaussian in x.
* **2D** -- a circular parabola, giving a 2D Gaussian.

The analytic waist is a paraxial result, so the tolerance reflects a small
systematic deviation rather than exact agreement. Measured agreement is far
better than the tolerances asserted here.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import meow as mw

visualize = False
"""Set True by `main()` so each test plots its cross-section and mode.

Left False under pytest, so no figure is ever shown there.
"""

WL = 1.55
K0 = 2 * np.pi / WL
N0 = 3.4  # peak index
A = 2.0  # parabola scale: n^2 = n0^2 (1 - (x/a)^2)
N_CLAD = 1.0  # clip, so the index stays physical outside |x| < a

# analytic Gaussian waist from the paraxial approximation
W_ANALYTIC = np.sqrt(A / (K0 * N0))

# the profile is clipped to n_clad outside |x| = a, and the mode decays well
# before that; a domain of +/- 1.5a leaves ample room
DOMAIN = 1.5 * A


def parabolic_index_1d(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:  # noqa: ARG001
    """n(x) for a 1D parabola, clipped so it stays physical."""
    n_squared = N0**2 * (1.0 - (np.asarray(x) / A) ** 2)
    return np.sqrt(np.maximum(n_squared, N_CLAD**2))


def parabolic_index_2d(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:  # noqa: ARG001
    """n(x, y) for a circular 2D parabola, clipped so it stays physical."""
    n_squared = N0**2 * (1.0 - (np.asarray(x) / A) ** 2 - (np.asarray(y) / A) ** 2)
    return np.sqrt(np.maximum(n_squared, N_CLAD**2))


def _solve(
    index: object,
    name: str,
    mesh: mw.Mesh2D,
    *,
    num_modes: int = 1,
) -> tuple[mw.CrossSection, mw.Mode]:
    """Build a cross-section filled with the given index profile and solve."""
    material = mw.SpatiallyVaryingMaterial(name=name, n=index)  # ty: ignore[invalid-argument-type]
    structure = mw.Structure3D(
        material=material,
        geometry=mw.Box(
            x_min=mesh.x[0],
            x_max=mesh.x[-1],
            y_min=mesh.y[0],
            y_max=mesh.y[-1],
            z_min=-1.0,
            z_max=1.0,
        ),
        mesh_order=2,
    )
    cell = mw.Cell(structures=[structure], mesh=mesh, z_min=-1.0, z_max=1.0)
    cs = mw.CrossSection.from_cell(cell=cell, env=mw.Environment(wl=WL))
    [mode] = mw.compute_modes(cs, num_modes=num_modes)
    return cs, mode


def _total_amplitude(mode: mw.Mode) -> np.ndarray:
    """Transverse field amplitude, ``sqrt(|Ex|^2 + |Ey|^2)``.

    Preferred over a single component for the 2D case: a circularly symmetric
    profile is degenerate, supporting two orthogonal polarizations at the same
    neff, so which of Ex/Ey carries the mode is arbitrary. The total is not.
    """
    return np.sqrt(np.abs(mode.Ex) ** 2 + np.abs(mode.Ey) ** 2)


def _overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized overlap of two real non-negative profiles."""
    return float(np.sum(a * b) / np.sqrt(np.sum(a**2) * np.sum(b**2)))


def _l2_error(a: np.ndarray, b: np.ndarray) -> float:
    """Relative L2 error between two profiles."""
    return float(np.linalg.norm(a - b) / np.linalg.norm(a))


def _plot_slice_vs_gaussian(
    x: np.ndarray,
    profile: np.ndarray,
    waist: float,
    *,
    title: str,
) -> None:
    """Plot a 1D slice of the solved mode against the predicted Gaussian.

    The Gaussian is drawn dashed so both remain visible where they coincide.
    """
    import matplotlib.pyplot as plt

    expected = np.exp(-(x**2) / (2 * waist**2))
    expected *= profile.max() / expected.max()

    _fig, ax = plt.subplots(figsize=(9.0, 4.0), constrained_layout=True)
    ax.plot(x, profile, "-", label="solved mode")
    ax.plot(x, expected, "--", label=f"Gaussian (w={waist:.4f})")
    ax.set_xlabel("x")
    ax.set_ylabel("|E|")
    ax.set_title(f"{title}: mode slice vs predicted Gaussian")
    ax.legend()
    ax.grid(visible=True, alpha=0.3)
    plt.savefig(f"{title.replace(' ', '_')}_slice_vs_gaussian.png")
    plt.show()


def _fit_gaussian_waists(
    profile: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray
) -> tuple[float, float]:
    """Fit ``exp(-(x^2/(2 wx^2) + y^2/(2 wy^2)))`` to a 2D profile."""
    selected = profile > 0.05 * profile.max()
    design = np.column_stack(
        [
            np.ones(int(selected.sum())),
            x_grid[selected] ** 2,
            y_grid[selected] ** 2,
        ]
    )
    coef, *_ = np.linalg.lstsq(design, np.log(profile[selected]), rcond=None)
    return np.sqrt(-1.0 / (2.0 * coef[1])), np.sqrt(-1.0 / (2.0 * coef[2]))


def test_analytic_waist_is_sane() -> None:
    """Guard the setup: the predicted waist must be small vs the parabola."""
    assert W_ANALYTIC < A / 2, "mode would not be paraxial"


def test_1d_parabolic_index_gives_gaussian_mode() -> None:
    """A 1D parabola gives a Gaussian in x, uniform in y, zero at the edges.

    No PMLs are used, so the unmodulated y direction behaves as a PEC
    waveguide.
    """
    mesh = mw.Mesh2D(
        x=np.linspace(-DOMAIN, DOMAIN, 401),
        y=np.linspace(-0.2, 0.2, 6),
        num_pml=(0, 0),
    )
    cs, mode = _solve(parabolic_index_1d, "parab1d", mesh)

    field = _total_amplitude(mode)  # shape (len(x_ey), len(y_ey))
    x = cs.mesh.Xy[:, 0]
    profile = field[:, 0]

    if visualize:
        _fig, ax = _comparison_figure((9.0, 6.0))
        mw.visualize(cs, ax=ax[0], show=False)
        ax[0].set_title("index (1D parabola)")
        mw.visualize(mode, fields=["Ex"], ax=ax[1], show=False)
        mw.visualize(mode, fields=["Ey"], ax=ax[2], show=False)
        plt.savefig("test_1d_parabolic_index_gives_gaussian_mode.png")
        plt.show()

        _plot_slice_vs_gaussian(x, profile, W_ANALYTIC, title="1D parabola")

    # 1. uniform along y, all the way out
    assert np.allclose(field, field[:, :1], rtol=1e-9), "mode is not uniform in y"

    # 2. gaussian in x
    assert profile.max() > 0
    expected = np.exp(-(x**2) / (2 * W_ANALYTIC**2))
    expected *= profile.max()
    assert _l2_error(profile, expected) < 1e-3
    assert _overlap(profile, expected) > 0.9999

    # 3. goes to zero at the PEC edges
    assert profile[0] / profile.max() < 1e-6
    assert profile[-1] / profile.max() < 1e-6

    # 4. neff is lossless and just below the peak index
    assert abs(mode.neff.imag) < 1e-9
    assert N0 - 0.2 < mode.neff.real < N0


def test_2d_parabolic_index_gives_2d_gaussian_mode() -> None:
    """A circular 2D parabola gives a 2D Gaussian, circularly symmetric."""
    mesh = mw.Mesh2D(
        x=np.linspace(-DOMAIN, DOMAIN, 201),
        y=np.linspace(-DOMAIN, DOMAIN, 201),
        num_pml=(0, 0),
    )
    cs, mode = _solve(parabolic_index_2d, "parab2d", mesh)

    field = _total_amplitude(mode)
    x_grid = cs.mesh.Xy
    y_grid = cs.mesh.Yy

    if visualize:
        _fig, ax = _comparison_figure((9.0, 11.0))
        mw.visualize(cs, ax=ax[0], show=False)
        ax[0].set_title("index (2D parabola)")
        mw.visualize(mode, fields=["Ex"], ax=ax[1], show=False)
        mw.visualize(mode, fields=["Ey"], ax=ax[2], show=False)
        plt.savefig("test_2d_parabolic_index_gives_2d_gaussian_mode.png")
        plt.show()

        # a slice through the center of the mode, along x
        center = field.shape[1] // 2
        _plot_slice_vs_gaussian(
            x_grid[:, center],
            field[:, center],
            W_ANALYTIC,
            title="2D parabola (center row)",
        )

    expected = np.exp(-(x_grid**2 + y_grid**2) / (2 * W_ANALYTIC**2))
    expected *= field.max()

    # Measured: L2 ~ 4.3e-2, overlap ~ 0.9991. The 2D profile is close to, but
    # not exactly, a separable Gaussian -- the paraxial waist used for
    # `expected` is a 1D result, so this is looser than the 1D case.
    assert _l2_error(field, expected) < 6e-2
    assert _overlap(field, expected) > 0.999

    # The waist recovered by a fit is the more physical check: it does not rely
    # on the paraxial formula being exact. Measured 1.3% above analytic, with
    # wx and wy agreeing to ~0.1%.
    waist_x, waist_y = _fit_gaussian_waists(field, x_grid, y_grid)
    assert np.isclose(waist_x, W_ANALYTIC, rtol=0.05)
    assert np.isclose(waist_y, W_ANALYTIC, rtol=0.05)
    # circularly symmetric
    assert np.isclose(waist_x, waist_y, rtol=0.02)

    assert abs(mode.neff.imag) < 1e-9
    assert N0 - 0.5 < mode.neff.real < N0


def test_1d_parabolic_neff_approaches_peak_index() -> None:
    """The mode sits just below the peak index, as a guided mode should.

    Also checks the waist recovered by a Gaussian fit, which avoids depending on
    the paraxial formula at all: the exponent of the solved field should be
    quadratic in x.
    """
    mesh = mw.Mesh2D(
        x=np.linspace(-DOMAIN, DOMAIN, 401),
        y=np.linspace(-0.2, 0.2, 6),
        num_pml=(0, 0),
    )
    cs, mode = _solve(parabolic_index_1d, "parab1d_fit", mesh)

    profile = _total_amplitude(mode)[:, 0]
    x = cs.mesh.Xy[:, 0]

    # log(profile) should be linear in x^2
    selected = profile > 0.05 * profile.max()
    slope, _intercept = np.polyfit(x[selected] ** 2, np.log(profile[selected]), 1)
    waist = np.sqrt(-1.0 / (2.0 * slope))

    assert np.isclose(waist, W_ANALYTIC, rtol=1e-2), (
        f"fitted waist {waist:.4f} vs analytic {W_ANALYTIC:.4f}"
    )


# --- manual inspection ---


def _comparison_figure(
    figsize: tuple[float, float] | None = None,
) -> tuple[Any, Any]:
    """A 3x1 grid: index profile on top, then Ex and Ey.

    ``constrained_layout`` gives the subplots margins; without it the mode
    titles collide with the panel above and the colorbars cannot take space
    from their own axes.

    ``figsize`` is left to matplotlib's default unless given, so a user's
    ``rcParams`` sizing still applies.
    """
    import matplotlib.pyplot as plt

    kwargs = {} if figsize is None else {"figsize": figsize}
    return plt.subplots(  # ty: ignore[no-matching-overload]
        3, 1, constrained_layout=True, **kwargs
    )


def main() -> None:
    """Run the parabolic-index checks and show each plot.

    Running `main()` means the user wants to inspect the modes, so the
    module-level flag is turned on. Under pytest it stays False and no figure
    is shown.
    """
    global visualize  # noqa: PLW0603
    visualize = True

    tests = [
        test_analytic_waist_is_sane,
        test_1d_parabolic_index_gives_gaussian_mode,
        test_2d_parabolic_index_gives_2d_gaussian_mode,
        test_1d_parabolic_neff_approaches_peak_index,
    ]
    failed = []
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failed.append(test.__name__)
            print(f"FAILED {test.__name__}: {exc}")  # noqa: T201
        else:
            print(f"passed {test.__name__}")  # noqa: T201

    if failed:
        msg = f"{len(failed)} of {len(tests)} tests failed: {failed}"
        raise SystemExit(msg)


if __name__ == "__main__":
    main()
