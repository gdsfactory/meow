"""Colorbar helpers shared by the Cell, CrossSection and Mode plots."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.ticker import Locator
from mpl_toolkits.axes_grid1 import make_axes_locatable


def colorbar_axes(ax: Any) -> Any:
    """Create the axes for a colorbar to the right of ``ax``, matching its height.

    Both options keep the bar as tall as ``ax`` (which, with a fixed aspect, may
    be shorter than its layout slot); they differ in who makes room.

    - With a layout engine (e.g. ``layout="constrained"``), an inset axes just
      outside ``ax``. The engine counts it as part of ``ax`` and makes room for
      the bar and its tick labels, also between neighbouring subplots.
    - Without one, nothing would make that room, and an inset would hang off
      the figure's edge. ``make_axes_locatable`` instead takes the bar's space
      out of ``ax`` itself. Its axes is invisible to a layout engine, which is
      why it is not used when there is one.
    """
    if ax.figure.get_layout_engine() is not None:
        return ax.inset_axes((1.03, 0.0, 0.05, 1.0))

    return make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)


class MaterialIndexLocator(Locator):
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
