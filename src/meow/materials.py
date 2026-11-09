"""Meow Materials."""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Literal, Self, TypeAlias, cast

import numpy as np
import pandas as pd
import tidy3d as td
from pydantic import Field, field_serializer, model_validator
from scipy.constants import c
from scipy.ndimage import map_coordinates
from tidy3d import material_library
from tidy3d.components.medium import AbstractMedium
from tidy3d.components.types import Axis as TdAxis
from tidy3d.material_library.material_library import VariantItem, VariantItemUniaxial

from meow.arrays import Dim, DType, NDArray
from meow.base_model import BaseModel
from meow.environment import Environment

ComponentType = Literal["x", "y", "z"]
"""Label representing a coordinate axis: 'x', 'y', or 'z'."""

component_str_to_int: dict[ComponentType, TdAxis] = {
    "x": 0,
    "y": 1,
    "z": 2,
}


class MaterialBase(BaseModel):
    """a `Material` defines the index of a `Structure3D` in an `Environment`."""

    name: str = Field(description="the name of the material")
    meta: dict[str, Any] = Field(
        default_factory=dict, description="metadata for the material"
    )

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        MATERIALS[self.name] = self
        return self

    def __call__(
        self, env: Environment, component: ComponentType | None = None
    ) -> float | complex:
        """Get the refractive index of the material.

        For the given environment, along the specified component ('x', 'y', or 'z').
        """
        msg = "Please use one of the Material child classes"
        raise NotImplementedError(msg)

    def _lumadd(self, sim: Any, env: Environment, unit: float) -> str:
        import matplotlib as mpl

        n = self(env)
        wl = np.asarray(env.wl * unit).ravel()
        eps = np.asarray(n).ravel() ** 2
        data = np.stack([c / wl, eps], 1)  # permittivity, not index

        if not sim.materialexists(self.name):
            sim.setmaterial(sim.addmaterial("Sampled data"), "name", self.name)
            color = np.asarray(
                self.meta.get("color") or mpl.colormaps["jet"](np.abs(eps) / 15.0)
            )
            sim.setmaterial(self.name, "color", color)

        sim.setmaterial(self.name, "sampled data", data)

        return self.name


class TidyMaterial(MaterialBase):
    """A material from the Tidy3D material library."""

    name: str = Field(description="The material name as also used by tidy3d")
    variant: str = Field(description="The material variant as also used by tidy3d")
    optical_axis: ComponentType | None = Field(
        default=None,
        description="""The optical axis (axis with extraordinary refractive index).
        Required for anisotropic materials. Can be None for isotropic materials""",
    )
    ignore_off_diagonal_permittivity: bool = Field(
        default=False,
        description="""Whether to ignore off-diagonal elements of permittivity tensor.
        If False (default), ValueError is raised when off-diagonal elements are present.
        If True, only the diagonal elements of the permittivity tensor are considered.
        Currently, only diagonal (along x, y, z axes) anisotropy is supported.""",
    )

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the TidyMaterial."""
        super().__init__(**kwargs)

        if self.name not in material_library:
            msg = (
                "Specified material name is invalid. "
                f"Use one of {material_library.keys()}"
            )
            raise ValueError(msg)
        _material = material_library[self.name]
        _variants = getattr(_material, "variants", {})
        if not _variants:
            msg = f"Tidy3D material '{self.name}' not supported."
            raise ValueError(msg)
        if self.variant not in _variants:
            _variant_options = list(_variants.keys())
            msg = f"Specified variant is invalid. Use one of {_variant_options}."
            raise ValueError(msg)

    def __call__(
        self, env: Environment, component: ComponentType | None = None
    ) -> float | complex:
        """Get the refractive index of the material.

        For the given environment, along the specified component ('x', 'y', or 'z').
        """
        if not isinstance(env, Environment):
            env = Environment(**env)
        _material = material_library[self.name]
        _variants = getattr(_material, "variants", None)
        if _variants is None:
            msg = f"Tidy3D material '{self.name}' not supported."
            raise ValueError(msg)

        variant = _material.variants[self.variant]

        if isinstance(variant, VariantItem):
            material = variant.medium
            eps = material.eps_comp(0, 0, td.C_0 / env.wl)
        elif isinstance(variant, VariantItemUniaxial):
            material = variant.medium
            if component not in ["x", "y", "z"]:
                msg = (
                    "Component must be specified for anisotropic materials. "
                    f"Must be 'x', 'y', or 'z', but it was {component}."
                )
                raise ValueError(msg)

            if self.optical_axis not in ["x", "y", "z"]:
                msg = (
                    "Optical axis must be specified for anisotropic materials. "
                    f"Must be 'x', 'y', or 'z', but it was {self.optical_axis}."
                )
                raise ValueError(msg)

            material_along_axis = material(component_str_to_int[self.optical_axis])
            if not self.ignore_off_diagonal_permittivity:
                self._ensure_no_off_diagonal(material_along_axis, freq=td.C_0 / env.wl)
            eps = material_along_axis.eps_comp(
                component_str_to_int[component],
                component_str_to_int[component],
                td.C_0 / env.wl,
            )
        else:
            msg = (
                f"Unsupported material type: {type(variant)}."
                " Supported types are VariantItem or VariantItemUniaxial."
            )
            raise TypeError(msg)

        return np.sqrt(eps).item()

    def _ensure_no_off_diagonal(
        self, tidy3d_medium: AbstractMedium, freq: float
    ) -> None:
        """Ensure the material has no off-diagonal permittivity components."""
        for i in range(3):
            for j in range(3):
                if i != j and tidy3d_medium.eps_comp(i, j, freq) != 0:  # ty: ignore[invalid-argument-type]
                    msg = (
                        f"Tidy3D material '{self.name}' variant '{self.variant}' "
                        "has off-diagonal permittivity components. Currently, "
                        "only diagonal (along x, y, z axes) anisotropy is supported."
                    )
                    raise ValueError(msg)


class DispersionModelBase(BaseModel, ABC):
    """A model returning a refractive index for a given `Environment`.

    Subclasses should call `self._check_wavelength(env)` before computing an index.
    It validates that `env.wl` lies within the model's
    `wavelength_lower_limit`..`wavelength_upper_limit` range, outside of which
    the fit is not defined; skipping the check would silently return an index
    extrapolated beyond the fit's validity.
    """

    wavelength_lower_limit: float | None = Field(
        default=None,
        description="lower limit (um) of the valid wavelength range, or None for "
        "no lower limit",
    )
    wavelength_upper_limit: float | None = Field(
        default=None,
        description="upper limit (um) of the valid wavelength range, or None for "
        "no upper limit",
    )

    @model_validator(mode="after")
    def _validate_wavelength_limits(self) -> Self:
        if (
            self.wavelength_lower_limit is not None
            and self.wavelength_upper_limit is not None
            and self.wavelength_lower_limit > self.wavelength_upper_limit
        ):
            msg = (
                "wavelength_lower_limit must not exceed wavelength_upper_limit. "
                f"Got {self.wavelength_lower_limit} > "
                f"{self.wavelength_upper_limit}."
            )
            raise ValueError(msg)
        return self

    def _check_wavelength(self, env: Environment) -> None:
        """Check that `env.wl` is within this model's valid wavelength range.

        Subclasses must call this as the first step of their `__call__`, so that
        the validity check is always performed.

        Raises:
            ValueError: if `env.wl` is outside the valid wavelength range.
        """
        lower = (
            0 if self.wavelength_lower_limit is None else self.wavelength_lower_limit
        )
        upper = (
            float("inf")
            if self.wavelength_upper_limit is None
            else self.wavelength_upper_limit
        )
        if env.wl <= 0:
            msg = (
                f"Environment wavelength {env.wl} um must be positive. "
                "Refractive index models are defined for positive wavelengths."
            )
            raise ValueError(msg)
        if not (lower <= env.wl <= upper):
            lo_str = "no lower limit" if self.wavelength_lower_limit is None else lower
            hi_str = "no upper limit" if self.wavelength_upper_limit is None else upper
            msg = (
                f"Environment wavelength {env.wl} um is outside the valid range "
                f"of this dispersion model ({lo_str} to {hi_str} um)."
            )
            raise ValueError(msg)

    @abstractmethod
    def __call__(self, env: Environment) -> float | complex:
        """Get the refractive index for the given environment.

        Implementations should call `self._check_wavelength(env)` first.
        """


class SellmeierModel(DispersionModelBase):
    """Sellmeier dispersion: n^2 = A + sum_i B_i wl^2 / (wl^2 - C_i).

    The wavelength is in micrometers (as in `Environment`), so the `C_i` are in
    micrometers squared. The temperature of the environment is ignored.
    """

    A: float = Field(default=1.0, description="constant term of n^2")
    B: tuple[float, ...] = Field(description="Sellmeier B coefficients (unitless)")
    C: tuple[float, ...] = Field(description="Sellmeier C coefficients (um^2)")

    @model_validator(mode="after")
    def _validate_lengths(self) -> Self:
        if len(self.B) != len(self.C):
            msg = (
                "B and C must have the same length. "
                f"Got {len(self.B)} and {len(self.C)}."
            )
            raise ValueError(msg)
        return self

    def __call__(self, env: Environment) -> float:
        """Get the refractive index for the given environment.

        Raises:
            ValueError: if the wavelength is outside the valid range (see
                `DispersionModelBase._check_wavelength`), or if n^2 < 0.
                Sellmeier models describe lossless media, so they cannot produce
                a complex index; a negative n^2 means the fit is not physical
                there.
        """
        self._check_wavelength(env)
        wl2 = env.wl**2
        n2 = self.A + sum(
            (b * wl2 / (wl2 - c) for b, c in zip(self.B, self.C, strict=True)),
            start=0.0,
        )
        if n2 < 0:
            msg = (
                f"Sellmeier model is not physical at wl={env.wl} (um): "
                f"n^2 = {n2} < 0. Sellmeier models describe lossless media and "
                "cannot represent absorption; use a Lorentz-Debye-Drude model "
                "for lossy media, or a wavelength within the fit's range."
            )
            raise ValueError(msg)
        return float(np.sqrt(n2))


class CauchyModel(DispersionModelBase):
    """Cauchy dispersion: n = A + B / wl^2 + C / wl^4.

    The wavelength is in micrometers (as in `Environment`), so B is in
    micrometers squared and C in micrometers to the fourth. The temperature of the
    environment is ignored.
    """

    A: float = Field(description="Cauchy A coefficient (unitless)")
    B: float = Field(default=0.0, description="Cauchy B coefficient (um^2)")
    C: float = Field(default=0.0, description="Cauchy C coefficient (um^4)")

    def __call__(self, env: Environment) -> float:
        """Get the refractive index for the given environment.

        Raises:
            ValueError: if the wavelength is outside the valid range (see
                `DispersionModelBase._check_wavelength`).
        """
        self._check_wavelength(env)
        wl2 = env.wl**2
        return self.A + self.B / wl2 + self.C / wl2**2


class IndexMaterial(MaterialBase):
    """A material with a constant refractive index."""

    n: complex = Field(description="the refractive index of the material")

    def __call__(
        self,
        env: Environment,  # noqa: ARG002
        component: ComponentType | None = None,  # noqa: ARG002
    ) -> float | complex:
        """Get the refractive index of the material.

        For the given environment, along the specified component ('x', 'y', or 'z').
        """
        return self.n


class AnisotropicIndexMaterial(MaterialBase):
    """A material with a constant anisotropic refractive index."""

    nx: complex = Field(description="refractive index along the x component")
    ny: complex = Field(description="refractive index along the y component")
    nz: complex = Field(description="refractive index along the z component")

    def __call__(
        self,
        env: Environment,  # noqa: ARG002
        component: ComponentType | None = None,
    ) -> float | complex:
        """Get the refractive index of the material.

        For the given environment, along the specified component ('x', 'y', or 'z').
        """
        if component == "x":
            result = self.nx
        elif component == "y":
            result = self.ny
        elif component == "z":
            result = self.nz
        else:
            msg = f"Component must be 'x', 'y', or 'z', but it was {component}."
            raise ValueError(msg)

        return result


class SpatiallyVaryingMaterial(MaterialBase):
    """A material whose refractive index is given by a user-supplied function.

    The function is called positionally as ``n(x, y, z)``, where ``x`` and ``y``
    are 2D coordinate arrays at the requested field component's staggered
    positions and ``z`` is a float or ``None``:

    - ``z`` is ``None`` when the cross-section was built without a known
      z-position (see ``CrossSection.z``). Each implementation decides how to
      handle this: a material that does not vary along z may ignore ``z``
      entirely, whereas a material that genuinely depends on z should raise.

    - The function may return a scalar (broadcast over the mesh) or an array
      matching the coordinate arrays.

    Either give a single isotropic callable ``n``, or all three of ``nx``,
    ``ny`` and ``nz`` (equivalently, a biaxial material), mirroring
    ``IndexMaterial`` and ``AnisotropicIndexMaterial`` respectively.
    """

    n: Callable[..., Any] | None = Field(
        default=None,
        description="isotropic refractive index as a function of (x, y, z)",
    )
    nx: Callable[..., Any] | None = Field(
        default=None,
        description="refractive index along x as a function of (x, y, z)",
    )
    ny: Callable[..., Any] | None = Field(
        default=None,
        description="refractive index along y as a function of (x, y, z)",
    )
    nz: Callable[..., Any] | None = Field(
        default=None,
        description="refractive index along z as a function of (x, y, z)",
    )

    @model_validator(mode="after")
    def _validate_callables(self) -> Self:
        """Check that the given index callables exist and are actually callable."""
        components = {"x": self.nx, "y": self.ny, "z": self.nz}
        given = {c: v for c, v in components.items() if v is not None}

        if self.n is not None and given:
            msg = (
                f"Material {self.name!r} must specify either an isotropic 'n' or "
                f"the per-axis 'nx'/'ny'/'nz', not both. Got n and {sorted(given)}."
            )
            raise ValueError(msg)
        if self.n is None and not given:
            msg = (
                f"Material {self.name!r} must specify either an isotropic 'n' or "
                "all of 'nx', 'ny' and 'nz'."
            )
            raise ValueError(msg)
        if given and set(given) != {"x", "y", "z"}:
            msg = (
                f"Material {self.name!r} must specify all of 'nx', 'ny' and 'nz'. "
                f"Got {sorted(given)}."
            )
            raise ValueError(msg)

        for label, func in [("n", self.n), *components.items()]:
            if func is not None and not callable(func):
                msg = (
                    f"Material {self.name!r} field {label!r} must be callable. "
                    f"Got {type(func).__name__}."
                )
                raise TypeError(msg)
        return self

    def _callable_for(self, component: ComponentType | None) -> Callable[..., Any]:
        """Return the index callable for the given component.

        Raises:
            ValueError: if ``component`` is not one of 'x', 'y', 'z' for an
                anisotropic material.
        """
        if self.n is not None:
            return self.n
        if component not in ("x", "y", "z"):
            msg = (
                "Component must be specified for anisotropic materials. "
                f"Must be 'x', 'y', or 'z', but it was {component}."
            )
            raise ValueError(msg)
        return cast(
            Callable[..., Any], {"x": self.nx, "y": self.ny, "z": self.nz}[component]
        )

    def __call__(
        self,
        env: Environment,
        component: ComponentType | None = None,
    ) -> float | complex:
        """Not supported: this material needs coordinates to be evaluated.

        This just raises NotImplementedError. Use `n_at` instead, which
        `CrossSection` calls with the mesh coordinates.
        """
        msg = (
            f"Material {self.name!r} is spatially varying and cannot be evaluated "
            "without coordinates. Use 'n_at(env, x, y, z, component=...)' instead."
        )
        raise NotImplementedError(msg)

    def n_at(
        self,
        env: Environment,  # noqa: ARG002
        x: NDArray,
        y: NDArray,
        z: float | None,
        component: ComponentType | None = None,
    ) -> NDArray:
        """Evaluate the refractive index at the given coordinates.

        If x or y are arrays, they must be broadcastable to the same shape. The
        returned refractive index will have a shape broadcasted to match x and y.

        Args:
            env: the environment. Not passed to the index callable, but
                overridden methods in subclasses may use it.
            x: x coordinates.
            y: y coordinates.
            z: z coordinate, or None if unknown.
            component: 'x', 'y' or 'z'; required for an anisotropic material.

        Returns:
            The refractive index, as an array broadcastable to ``x``.
        """
        func = self._callable_for(component)
        return np.asarray(func(x, y, z))

    @field_serializer("n", "nx", "ny", "nz")
    def _serialize_callable(self, value: Callable[..., Any] | None) -> Any:
        """Raise a clear error instead of orjson's generic message.

        Serialization already fails loudly without this (``orjson`` cannot
        encode a function); this only improves the message, naming both the
        material and the callable that cannot be encoded.
        """
        if value is None:
            return None
        msg = (
            f"Material {self.name!r} is spatially varying and cannot be serialized: "
            f"its refractive index is the callable {value!r}, which has no "
            "serialized representation. Reconstruct it from source instead."
        )
        raise TypeError(msg)

    def __reduce__(self) -> Any:
        """Raise: pickling (e.g. `Mode.save`) cannot roundtrip a callable."""
        msg = (
            f"Material {self.name!r} is spatially varying and cannot be pickled: "
            "its refractive index is a user-supplied Python callable, which is "
            "typically a closure and cannot be reconstructed."
        )
        raise TypeError(msg)


class SampledMaterial(MaterialBase):
    """A material with a sampled refractive index."""

    # TODO: use the new sax xarray interpolation

    n: Annotated[NDArray, Dim(1), DType("complex128")] = Field(
        description="the complex refractive index of the material"
    )
    params: dict[str, Annotated[NDArray, Dim(1), DType("float64")]] = Field(
        description="the wavelength over which the refractive index is defined."
    )

    @staticmethod
    def _validate_1d(name: str, arr: np.ndarray) -> np.ndarray:
        if arr.ndim != 1:
            msg = f"{name} should be 1D. Got a {arr.ndim}D array."
            raise ValueError(msg)
        return arr

    @model_validator(mode="after")
    def _validate_params_length(self: Self) -> Self:
        Ln = self.n.shape[0]
        for p, v in self.params.items():
            Lp = v.shape[0]
            if Lp != Ln:
                msg = (
                    f"length of parameter array {p} does not match length "
                    f"of refractive index array n. \n {Lp} != {Ln}"
                )
                raise ValueError(msg)
        return self

    @classmethod
    def from_path(cls, path: str, meta: dict | None = None) -> Self:
        """Create a SampledMaterial from a CSV file."""
        path = _validate_path(path)
        name = re.sub(r"\.csv$", "", os.path.split(path)[-1])
        df = pd.read_csv(path)
        return cls.from_df(name, df, meta=meta)

    @classmethod
    def from_df(cls, name: str, df: pd.DataFrame, meta: dict | None = None) -> Self:
        """Create a SampledMaterial from a DataFrame."""
        meta = meta or {}

        nr = df["nr"].to_numpy()
        ni = np.zeros_like(nr) if "ni" not in df else df["ni"].to_numpy()
        n = nr + 1j * ni

        columns = [c for c in df.columns if c not in ["nr", "ni"]]
        params = {c: np.asarray(df[c].values, dtype=np.float64) for c in columns}

        return cls(name=name, params=params, n=n, meta=meta)

    def __call__(
        self,
        env: Environment,
        component: ComponentType | None = None,  # noqa: ARG002
    ) -> float | complex:
        """Get the refractive index of the material.

        For the given environment, along the specified component ('x', 'y', or 'z').
        """
        if not isinstance(env, Environment):
            env = Environment(**env)
        df = pd.DataFrame({**self.params, "nr": np.real(self.n), "ni": np.imag(self.n)})
        data, params, strings = _to_ndgrid(df, wl_key="wl")
        result, axs, pos = _evaluate_general_corner_model(
            data,
            params,
            strings,
            **{k: np.atleast_1d(v) for k, v in env.model_dump().items()},
        )
        nr = result.take(pos["targets"]["nr"], axs["targets"])
        ni = result.take(pos["targets"]["ni"], axs["targets"])
        n = nr + 1j * ni
        return n.item()


class SampledAnisotropicMaterial(MaterialBase):
    """A material with a sampled, diagonally anisotropic refractive index.

    The refractive index along each of the x, y and z axes is sampled
    independently over the same parameters (e.g. the wavelength). Equivalently,
    it is a biaxial material.
    """

    n: dict[ComponentType, Annotated[NDArray, Dim(1), DType("complex128")]] = Field(
        description="the complex refractive index along each of 'x', 'y' and 'z'"
    )
    params: dict[str, Annotated[NDArray, Dim(1), DType("float64")]] = Field(
        description="the parameters (e.g. wavelength) over which n is defined."
    )

    @model_validator(mode="after")
    def _validate_params_length(self: Self) -> Self:
        if set(self.n) != {"x", "y", "z"}:
            msg = f"n must have keys 'x', 'y' and 'z'. Got {sorted(self.n)}."
            raise ValueError(msg)
        for comp, n in self.n.items():
            for p, v in self.params.items():
                if v.shape[0] != n.shape[0]:
                    msg = (
                        f"length of parameter array {p} does not match length "
                        f"of refractive index array n[{comp!r}]. \n"
                        f"{v.shape[0]} != {n.shape[0]}"
                    )
                    raise ValueError(msg)
        return self

    @classmethod
    def from_path(
        cls,
        path: str,
        axis_num_to_dir: dict[int, ComponentType],
        meta: dict | None = None,
    ) -> Self:
        """Create an SampledAnisotropicMaterial from a CSV file.

        The CSV has columns `wl,n1r,n1i,n2r,n2i,n3r,n3i`. See `from_df`.
        """
        path = _validate_path(path)
        name = re.sub(r"\.csv$", "", os.path.split(path)[-1])
        df = pd.read_csv(path)
        return cls.from_df(name, df, axis_num_to_dir, meta=meta)

    @classmethod
    def from_df(
        cls,
        name: str,
        df: pd.DataFrame,
        axis_num_to_dir: dict[int, ComponentType],
        meta: dict | None = None,
    ) -> Self:
        """Create an SampledAnisotropicMaterial from a DataFrame.

        Args:
            name: the name of the material.
            df: dataframe with columns `wl`, `n{i}r` and (optionally) `n{i}i`
                for i in 1, 2, 3 (plus any other parameter columns).
            axis_num_to_dir: maps the axis numbers 1, 2, 3 used in the columns
                to the directions 'x', 'y', 'z'. Must be a bijection.
            meta: optional metadata.
        """
        meta = meta or {}
        if set(axis_num_to_dir) != {1, 2, 3} or set(axis_num_to_dir.values()) != {
            "x",
            "y",
            "z",
        }:
            msg = (
                "axis_num_to_dir must map each of 1, 2, 3 to a unique one of "
                f"'x', 'y', 'z'. Got {axis_num_to_dir}."
            )
            raise ValueError(msg)

        n = {}
        for i, direction in axis_num_to_dir.items():
            nr = df[f"n{i}r"].to_numpy()
            ni = df[f"n{i}i"].to_numpy() if f"n{i}i" in df else np.zeros_like(nr)
            n[direction] = nr + 1j * ni

        used = {f"n{i}{p}" for i in (1, 2, 3) for p in "ri"}
        params = {
            c: np.asarray(df[c].values, dtype=np.float64)
            for c in df.columns
            if c not in used
        }
        return cls(name=name, params=params, n=n, meta=meta)

    def __call__(
        self,
        env: Environment,
        component: ComponentType | None = None,
    ) -> float | complex:
        """Get the refractive index of the material.

        For the given environment, along the specified component ('x', 'y', or 'z').
        """
        if component not in ("x", "y", "z"):
            msg = f"Component must be 'x', 'y', or 'z', but it was {component}."
            raise ValueError(msg)
        if not isinstance(env, Environment):
            env = Environment(**env)
        n = self.n[component]
        df = pd.DataFrame({**self.params, "nr": np.real(n), "ni": np.imag(n)})
        data, params, strings = _to_ndgrid(df, wl_key="wl")
        result, axs, pos = _evaluate_general_corner_model(
            data,
            params,
            strings,
            **{k: np.atleast_1d(v) for k, v in env.model_dump().items()},
        )
        nr = result.take(pos["targets"]["nr"], axs["targets"])
        ni = result.take(pos["targets"]["ni"], axs["targets"])
        return (nr + 1j * ni).item()


class SampledUniaxialMaterial(SampledAnisotropicMaterial):
    """A material with a sampled, uniaxial refractive index.

    The extraordinary index applies along the `optical_axis` given at
    construction, and the ordinary index along the other two axes.
    """

    @classmethod
    def from_path(  # ty: ignore[invalid-method-override]
        cls,
        path: str,
        optical_axis: ComponentType,
        meta: dict | None = None,
    ) -> Self:
        """Create a SampledUniaxialMaterial from a CSV file.

        The CSV has columns `wl,nor,noi,ner,nei`. See `from_df`.
        """
        path = _validate_path(path)
        name = re.sub(r"\.csv$", "", os.path.split(path)[-1])
        df = pd.read_csv(path)
        return cls.from_df(name, df, optical_axis, meta=meta)

    @classmethod
    def from_df(  # ty: ignore[invalid-method-override]
        cls,
        name: str,
        df: pd.DataFrame,
        optical_axis: ComponentType,
        meta: dict | None = None,
    ) -> Self:
        """Create a SampledUniaxialMaterial from a DataFrame.

        Args:
            name: the name of the material.
            df: dataframe with columns `wl`, `nor`, `ner` and (optionally)
                `noi`, `nei` (ordinary / extraordinary, real / imaginary).
            optical_axis: the optical axis ('x', 'y' or 'z'), i.e. the axis
                along which the extraordinary index applies.
            meta: optional metadata.
        """
        if optical_axis not in ("x", "y", "z"):
            msg = f"optical_axis must be 'x', 'y' or 'z'. Got {optical_axis!r}."
            raise ValueError(msg)
        nor = df["nor"].to_numpy()
        ner = df["ner"].to_numpy()
        noi = df["noi"].to_numpy() if "noi" in df else np.zeros_like(nor)
        nei = df["nei"].to_numpy() if "nei" in df else np.zeros_like(ner)
        no, ne = nor + 1j * noi, ner + 1j * nei
        n = {d: (ne if d == optical_axis else no) for d in ("x", "y", "z")}
        params = {
            c: np.asarray(df[c].values, dtype=np.float64)
            for c in df.columns
            if c not in ("nor", "noi", "ner", "nei")
        }
        return cls(name=name, params=params, n=n, meta=meta or {})


DispersionModel: TypeAlias = DispersionModelBase | SellmeierModel | CauchyModel
"""A `SellmeierModel`, a `CauchyModel`, or any custom `DispersionModelBase`.

`DispersionModelBase` is included so that user-defined subclasses are accepted.
The concrete models are listed as well so that pydantic can serialize and
deserialize them: a field annotated with the bare base class only knows the
base class' fields, so dumping would silently drop the subclass fields.

Note:
    A user-defined subclass is accepted, but does not roundtrip (serialize/
    deserialize): it is not part of the union's known models, so serializing
    drops its fields and deserializing fails with a `ValidationError` rather
    than silently reconstructing the wrong model.
"""


class DispersiveMaterial(MaterialBase):
    """An isotropic material defined by a `DispersionModel`."""

    dispersion_model: DispersionModel = Field(
        description="the dispersion model, applied along all three axes"
    )

    def __call__(
        self,
        env: Environment,
        component: ComponentType | None = None,  # noqa: ARG002
    ) -> float | complex:
        """Get the refractive index of the material.

        For the given environment, along the specified component ('x', 'y', or 'z').
        """
        if not isinstance(env, Environment):
            env = Environment(**env)
        return self.dispersion_model(env)


class UniaxialDispersiveMaterial(MaterialBase):
    """A uniaxial material defined by two `DispersionModel`s."""

    ordinary_dispersion_model: DispersionModel = Field(
        description="the dispersion model of the ordinary index"
    )
    extraordinary_dispersion_model: DispersionModel = Field(
        description="the dispersion model of the extraordinary index"
    )
    optical_axis: ComponentType = Field(
        description="the optical axis ('x', 'y' or 'z'): the axis along which the "
        "extraordinary index applies"
    )

    def __call__(
        self, env: Environment, component: ComponentType | None = None
    ) -> float | complex:
        """Get the refractive index of the material.

        For the given environment, along the specified component ('x', 'y', or 'z').
        """
        if component not in ("x", "y", "z"):
            msg = f"Component must be 'x', 'y', or 'z', but it was {component}."
            raise ValueError(msg)
        if not isinstance(env, Environment):
            env = Environment(**env)
        if component == self.optical_axis:
            return self.extraordinary_dispersion_model(env)
        return self.ordinary_dispersion_model(env)


class AnisotropicDispersiveMaterial(MaterialBase):
    """A material with a different `DispersionModel` along each of x, y and z.

    Equivalently, a biaxial material.
    """

    dispersion_model_x: DispersionModel = Field(description="model along x")
    dispersion_model_y: DispersionModel = Field(description="model along y")
    dispersion_model_z: DispersionModel = Field(description="model along z")

    def __call__(
        self, env: Environment, component: ComponentType | None = None
    ) -> float | complex:
        """Get the refractive index of the material.

        For the given environment, along the specified component ('x', 'y', or 'z').
        """
        if component not in ("x", "y", "z"):
            msg = f"Component must be 'x', 'y', or 'z', but it was {component}."
            raise ValueError(msg)
        if not isinstance(env, Environment):
            env = Environment(**env)
        model = {
            "x": self.dispersion_model_x,
            "y": self.dispersion_model_y,
            "z": self.dispersion_model_z,
        }[component]
        return model(env)


Material: TypeAlias = (
    IndexMaterial
    | AnisotropicIndexMaterial
    | SpatiallyVaryingMaterial
    | SampledMaterial
    | SampledAnisotropicMaterial
    | SampledUniaxialMaterial
    | DispersiveMaterial
    | UniaxialDispersiveMaterial
    | AnisotropicDispersiveMaterial
    | TidyMaterial
)
"""A material: `IndexMaterial`, `AnisotropicIndexMaterial`,
`SpatiallyVaryingMaterial`, `SampledMaterial`, `SampledAnisotropicMaterial`,
`SampledUniaxialMaterial`, `DispersiveMaterial`, `UniaxialDispersiveMaterial`,
`AnisotropicDispersiveMaterial`, or `TidyMaterial`."""
Materials: TypeAlias = list[Material]
"""A list of `Material` objects."""
MATERIALS: dict[str, MaterialBase] = {}


def _to_ndgrid(df: pd.DataFrame, wl_key: str = "wl") -> tuple[Any, Any, Any]:
    """Convert stacked data to hypercube data.

    Args:
        df: the dataframe to convert to hypercube data
        wl_key: which key to use to sort. Options: 'wl' or 'f'.
        Using 'f' yields better interpolation results as most optical
        phenomina are more linear in 'f' than in 'wl'.

    Note:
        This only works if the number of other parameters (e.g. # wls) stays consistent
        for each corner.
    """
    df = df.copy()
    if wl_key not in ["f", "wl"]:
        msg = f"Unsupported wl_key. Valid choices: 'wl', 'f'. Got: {wl_key}."
        raise ValueError(msg)
    df["f"] = c / df["wl"].to_numpy()
    value_columns = [c for c in ["nr", "ni"] if c in df.columns]
    param_columns = [c for c in df.columns if c not in [*value_columns, "wl", "f"]]
    param_columns.append(wl_key)
    param_columns = [c for c in param_columns if c in df.columns]
    df_sel = cast(pd.DataFrame, df[param_columns + value_columns])
    df = _sort_rows(df_sel, not_by=tuple(value_columns), wl_key=wl_key)
    params: dict = {c: df[c].unique() for c in param_columns}
    data = df[value_columns].to_numpy()
    data = data.reshape(*(v.shape[0] for v in params.values()), len(value_columns))
    data = np.asarray(data)
    params["targets"] = np.array(value_columns, dtype=object)
    params, strings = _extract_strings(params)
    return data, params, strings


def _evaluate_general_corner_model(
    data: Any,
    params: Any,
    strings: Any,
    /,
    **kwargs: Any,
) -> tuple[np.ndarray, dict[str, int], dict[str, dict[str, int]]]:
    given_params = {k: kwargs.get(k, np.asarray(v).mean()) for k, v in params.items()}
    given_strings = {
        p: (list(v.values()) if p not in kwargs else [v[vv] for vv in kwargs[p]])
        for p, v in strings.items()
    }
    string_locations = {k: i for i, k in enumerate(given_params) if k in given_strings}
    param_locations = {
        k: i for i, k in enumerate(given_params) if k not in given_strings
    }
    data = np.transpose(
        data, axes=[*param_locations.values(), *string_locations.values()]
    )
    _, string_shape = (
        data.shape[: len(param_locations)],
        data.shape[len(param_locations) :],
    )
    given_params = {k: v for k, v in given_params.items() if k not in given_strings}
    num_params = len(given_params)
    data = _downselect(data, given_strings.values())
    param_shape, string_shape = data.shape[:num_params], data.shape[num_params:]
    data = data.reshape(*param_shape, -1)
    given_param_values = [np.asarray(v) for v in given_params.values()]
    stacked_params = np.stack(np.broadcast_arrays(*given_param_values), 0)
    coords = _get_coordinates(
        [v for k, v in params.items() if k in given_params], stacked_params
    )
    result = _map_coordinates(data, coords)
    axs = {k: i for i, k in enumerate(given_strings)}
    rev_strings = {k: {vv: kk for kk, vv in v.items()} for k, v in strings.items()}
    pos = {
        k: {rev_strings[k][vv]: i for i, vv in enumerate(v)}
        for k, v in given_strings.items()
    }
    return result.reshape(*string_shape, *result.shape[1:]), axs, pos


def _downselect(data: np.ndarray, idxs_list: Any) -> np.ndarray:
    for i, idxs in enumerate(reversed(idxs_list), start=1):
        data = data.take(np.asarray(idxs, dtype=int), axis=-i)
    return data


# @partial(jax.vmap, in_axes=(-1, None), out_axes=0)
# def _map_coordinates(input, coordinates):
#    return jax.scipy.ndimage.map_coordinates(input, coordinates, 1, mode="nearest")


def _map_coordinates(data: np.ndarray, coordinates: Any) -> np.ndarray:
    result = []
    for i in range(data.shape[-1]):
        current_result = map_coordinates(
            data[..., i], coordinates, order=1, mode="nearest"
        )
        result.append(current_result)
    return np.stack(result, 0)


def _get_coordinate(arr1d: np.ndarray, value: np.ndarray) -> np.ndarray:
    return np.interp(value, arr1d, np.arange(arr1d.shape[0]))


def _get_coordinates(arrs1d: list[np.ndarray], values: np.ndarray) -> np.ndarray:
    # don't use vmap as arrays in arrs1d could have different shapes...
    return np.array(
        [_get_coordinate(a, v) for a, v in zip(arrs1d, values, strict=False)]
    )


def _extract_strings(
    params: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, int]]]:
    string_map = {}
    new_params = {}
    for k, v in list(params.items()):
        if v.dtype == object:  # probably string array!
            string_map[k] = {s: i for i, s in enumerate(v)}
            new_params[k] = np.array(list(string_map[k].values()), dtype=int)
        else:
            new_params[k] = np.array(v, dtype=float)
    return new_params, string_map


def _sort_rows(
    df: pd.DataFrame, not_by: tuple[str, ...] = ("nr", "ni"), wl_key: str = "wl"
) -> pd.DataFrame:
    not_by = ("wl", "f", *not_by)
    by = [c for c in df.columns if c not in not_by] + [wl_key]
    return df.sort_values(by=by).reset_index(drop=True)


def _validate_path(path: str) -> str:
    if not path.endswith(".csv"):
        path = f"{path}.csv"
    path_parts = path.replace("\\", "/").split("/")
    if not Path(path).exists() and len(path_parts) == 1:
        lib_path = Path(__file__).resolve().parent
        path = str(lib_path / "assets" / path)
    if not Path(path).exists():
        msg = f"Material file {path!r} not found."
        raise FileNotFoundError(msg)
    return os.path.relpath(path, start=Path.cwd())


# NOTE: The materials defined below are not meant to be a comprehensive library
# of materials. They are examples of how to use the different material options
# in MEOW.

silicon = SampledMaterial.from_path(
    path="Si",
    meta={"color": (0.9, 0, 0, 0.9)},
)
silicon_oxide = SampledMaterial.from_path(
    path="SiO2",
    meta={"color": (0.9, 0.9, 0.9, 0.9)},
)

silicon_nitride = TidyMaterial(
    name="Si3N4",
    variant="Luke2015PMLStable",
    meta={"color": (0, 0.9, 0.9, 0.9)},
)

# Anisotropic LiNbO3 from the Tidy3D material library, with the optical
# axis along x, y or z.
lithium_niobate_optical_x = TidyMaterial(
    name="LiNbO3",
    variant="Zelmon1997",
    optical_axis="x",
    meta={"color": (0.9, 0.6, 0.1, 0.9)},
)
lithium_niobate_optical_y = TidyMaterial(
    name="LiNbO3",
    variant="Zelmon1997",
    optical_axis="y",
    meta={"color": (0.9, 0.6, 0.1, 0.9)},
)
lithium_niobate_optical_z = TidyMaterial(
    name="LiNbO3",
    variant="Zelmon1997",
    optical_axis="z",
    meta={"color": (0.9, 0.6, 0.1, 0.9)},
)

indium_phosphide_sellmeier_coefficients = {
    "A": 7.255,
    "B": (2.316, 2.765),
    "C": (0.6263**2, 32.935**2),
}

indium_phosphide = DispersiveMaterial(
    name="InP",
    dispersion_model=SellmeierModel(
        wavelength_lower_limit=0.95,
        wavelength_upper_limit=10,
        **indium_phosphide_sellmeier_coefficients,
    ),
    meta={"color": (0.7, 0.7, 0.3, 0.9)},
)

aluminum_nitride_sellmeier_coefficients_ordinary = {
    "A": 3.1399,
    "B": (1.3786, 3.861),
    "C": (0.1715**2, 15.03**2),
}
aluminum_nitride_sellmeier_coefficients_extraordinary = {
    "A": 3.0729,
    "B": (1.6173, 4.139),
    "C": (0.1746**2, 15.03**2),
}
aluminum_nitride_sellmeier_validity_range = {
    "wavelength_lower_limit": 0.22,
    "wavelength_upper_limit": 5.0,
}

aluminum_nitride_optical_x = UniaxialDispersiveMaterial(
    name="AlN",
    ordinary_dispersion_model=SellmeierModel(
        **aluminum_nitride_sellmeier_coefficients_ordinary,
        **aluminum_nitride_sellmeier_validity_range,
    ),
    extraordinary_dispersion_model=SellmeierModel(
        **aluminum_nitride_sellmeier_coefficients_extraordinary,
        **aluminum_nitride_sellmeier_validity_range,
    ),
    optical_axis="x",
    meta={"color": (0.3, 0.7, 0.5, 0.9)},
)
"""Aluminum nitride with optical axis (axis associated with extraordinary index) along
x. Refractive indices for ordinary and extraordinary axes are calculated from a
Sellmeier model."""

aluminum_nitride_optical_y = UniaxialDispersiveMaterial(
    name="AlN",
    ordinary_dispersion_model=SellmeierModel(
        **aluminum_nitride_sellmeier_coefficients_ordinary,
        **aluminum_nitride_sellmeier_validity_range,
    ),
    extraordinary_dispersion_model=SellmeierModel(
        **aluminum_nitride_sellmeier_coefficients_extraordinary,
        **aluminum_nitride_sellmeier_validity_range,
    ),
    optical_axis="y",
    meta={"color": (0.3, 0.7, 0.5, 0.9)},
)
"""Aluminum nitride with optical axis (axis associated with extraordinary index) along
y. Refractive indices for ordinary and extraordinary axes are calculated from a
Sellmeier model."""

aluminum_nitride_optical_z = UniaxialDispersiveMaterial(
    name="AlN",
    ordinary_dispersion_model=SellmeierModel(
        **aluminum_nitride_sellmeier_coefficients_ordinary,
        **aluminum_nitride_sellmeier_validity_range,
    ),
    extraordinary_dispersion_model=SellmeierModel(
        **aluminum_nitride_sellmeier_coefficients_extraordinary,
        **aluminum_nitride_sellmeier_validity_range,
    ),
    optical_axis="z",
    meta={"color": (0.3, 0.7, 0.5, 0.9)},
)
"""Aluminum nitride with optical axis (axis associated with extraordinary index) along
z. Refractive indices for ordinary and extraordinary axes are calculated from a
Sellmeier model."""


lithium_tantalate_optical_x = SampledUniaxialMaterial.from_path(
    "lithium_tantalate",
    optical_axis="x",
    meta={"color": (0.2, 0.4, 0.9, 0.9)},
)
"""Lithium tantalate with the optical axis (axis associated with
extraordinary index) along x axis.
LiTaO3 data source (Table IV, hexagonal crystals, 0.45-4.0 um):
Bond, W. L. "Measurement of the refractive indices of several crystals."
Journal of Applied Physics 36.5 (1965): 1674-1677."""

lithium_tantalate_optical_y = SampledUniaxialMaterial.from_path(
    "lithium_tantalate",
    optical_axis="y",
    meta={"color": (0.2, 0.4, 0.9, 0.9)},
)
"""Lithium tantalate with the optical axis (axis associated with
extraordinary index) along y axis.
LiTaO3 data source (Table IV, hexagonal crystals, 0.45-4.0 um):
Bond, W. L. "Measurement of the refractive indices of several crystals."
Journal of Applied Physics 36.5 (1965): 1674-1677."""

lithium_tantalate_optical_z = SampledUniaxialMaterial.from_path(
    "lithium_tantalate",
    optical_axis="z",
    meta={"color": (0.2, 0.4, 0.9, 0.9)},
)
"""Lithium tantalate with the optical axis (axis associated with
extraordinary index) along z axis.
LiTaO3 data source (Table IV, hexagonal crystals, 0.45-4.0 um):
Bond, W. L. "Measurement of the refractive indices of several crystals."
Journal of Applied Physics 36.5 (1965): 1674-1677."""


lithium_triborate = SampledAnisotropicMaterial.from_path(
    "lithium_triborate",
    axis_num_to_dir={1: "x", 2: "y", 3: "z"},
    meta={"color": (0.6, 0.2, 0.8, 0.9)},
)
"""LiB3O5 (LBO) data source (Table 3, principal refractive indices n_x, n_y, n_z,
0.2537-1.0642 um):
Chen, Chuangtian, et al. "New nonlinear-optical crystal: LiB3O5."
Journal of the Optical Society of America B 6.4 (1989): 616-621."""


# KTP (KTiOPO4) is biaxial, with a different Sellmeier equation along each of x, y
# and z. The paper writes each equation as n^2 = A + sum_i B_i / (wl^2 - C_i),
# whereas SellmeierModel uses n^2 = A + sum_i B_i' wl^2 / (wl^2 - C_i). The two
# are identical with B_i' = B_i / C_i and A' = A - sum_i B_i / C_i, since
# B / (wl^2 - C) = (B / C) * (wl^2 / (wl^2 - C) - 1).
ktp_sellmeier_coefficients_x = {
    "A": 3.29100 - (0.04140 / 0.03978 + 9.35522 / 31.45571),
    "B": (0.04140 / 0.03978, 9.35522 / 31.45571),
    "C": (0.03978, 31.45571),
}
ktp_sellmeier_coefficients_y = {
    "A": 3.45018 - (0.04341 / 0.04597 + 16.98825 / 39.43799),
    "B": (0.04341 / 0.04597, 16.98825 / 39.43799),
    "C": (0.04597, 39.43799),
}
ktp_sellmeier_coefficients_z = {
    "A": 4.59423 - (0.06206 / 0.04763 + 110.80672 / 86.12171),
    "B": (0.06206 / 0.04763, 110.80672 / 86.12171),
    "C": (0.04763, 86.12171),
}
ktp_sellmeier_validity_range = {
    "wavelength_lower_limit": 0.43,
    "wavelength_upper_limit": 3.54,
}

ktp = AnisotropicDispersiveMaterial(
    name="KTP",
    dispersion_model_x=SellmeierModel(
        **ktp_sellmeier_coefficients_x,
        **ktp_sellmeier_validity_range,
    ),
    dispersion_model_y=SellmeierModel(
        **ktp_sellmeier_coefficients_y,
        **ktp_sellmeier_validity_range,
    ),
    dispersion_model_z=SellmeierModel(
        **ktp_sellmeier_coefficients_z,
        **ktp_sellmeier_validity_range,
    ),
    meta={"color": (0.4, 0.8, 0.8, 0.9)},
)
"""Potassium titanyl phosphate (KTP), a biaxial crystal, with refractive indices
n_x, n_y, n_z from Sellmeier equations (valid from 0.43 to 3.54 um) fit to
flux-grown crystals at 20 C. The Sellmeier equations ignore temperature.
Kato, Kiyoshi, and Eiko Takaoka. "Sellmeier and thermo-optic dispersion formulas
for KTP." Applied Optics 41.24 (2002): 5040-5044."""
