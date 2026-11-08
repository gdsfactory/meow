"""Meow Materials."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated, Any, Literal, Self, TypeAlias, cast

import numpy as np
import pandas as pd
import tidy3d as td
from pydantic import Field, model_validator
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
    ) -> np.ndarray:
        """Get the refractive index of the material.

        For the given environment, along the specified component ('x', 'y', or 'z').
        """
        msg = "Please use one of the Material child classes"
        raise NotImplementedError(msg)

    def _lumadd(self, sim: Any, env: Environment, unit: float) -> str:
        import matplotlib as mpl

        n = self(env)
        wl = np.asarray(env.wl * unit, dtype=complex).ravel()
        eps = np.asarray(n, dtype=complex).ravel() ** 2
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
    ) -> np.ndarray:
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

        return np.sqrt(eps)

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


class IndexMaterial(MaterialBase):
    """A material with a constant refractive index."""

    n: complex = Field(description="the refractive index of the material")

    def __call__(
        self,
        env: Environment,  # noqa: ARG002
        component: ComponentType | None = None,  # noqa: ARG002
    ) -> np.ndarray:
        """Get the refractive index of the material.

        For the given environment, along the specified component ('x', 'y', or 'z').
        """
        return np.squeeze(np.asarray(self.n, dtype=np.complex128))


class AnisotropicIndexMaterial(MaterialBase):
    """A material with a constant anisotropic refractive index."""

    nx: complex = Field(description="refractive index along the x component")
    ny: complex = Field(description="refractive index along the y component")
    nz: complex = Field(description="refractive index along the z component")

    def __call__(
        self,
        env: Environment,  # noqa: ARG002
        component: ComponentType | None = None,
    ) -> np.ndarray:
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

        return np.squeeze(np.asarray(result, dtype=np.complex128))


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
    ) -> np.ndarray:
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
        return np.squeeze(n)


Material: TypeAlias = (
    IndexMaterial | AnisotropicIndexMaterial | SampledMaterial | TidyMaterial
)
"""A material: `IndexMaterial`, `AnisotropicIndexMaterial`,
`SampledMaterial`, or `TidyMaterial`."""
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
