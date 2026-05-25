"""Default mode solver."""

from meow.fde.meep import compute_modes_meep
from meow.fde.tidy3d import compute_modes_tidy3d

compute_modes = compute_modes_tidy3d
