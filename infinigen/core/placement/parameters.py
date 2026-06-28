from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, ClassVar, Self, TypeVar

import bpy
import numpy as np
from annotated_types import Ge, Le
from pydantic import BaseModel, ConfigDict

from infinigen.core.util.math import FixedSeed, int_hash

TParams = TypeVar("TParams", bound="AssetParameters")


def _field_bounds(field_info: Any) -> tuple[float, float] | None:
    ge = le = None
    for meta in field_info.metadata:
        if isinstance(meta, Ge):
            ge = meta.ge
        elif isinstance(meta, Le):
            le = meta.le
    if ge is not None and le is not None:
        return float(ge), float(le)
    return None


def _field_extra(field_info: Any) -> dict:
    extra = field_info.json_schema_extra
    return extra if isinstance(extra, dict) else {}


def _field_kind(field_info: Any) -> str | None:
    kind = _field_extra(field_info).get("kind")
    return kind if isinstance(kind, str) else None


def _field_choices(field_info: Any) -> list[Any]:
    return list(_field_extra(field_info).get("choices", []))


def _is_editable(field_info: Any) -> bool:
    extra = field_info.json_schema_extra
    if isinstance(extra, dict):
        return bool(extra.get("editable", True))
    return True


class AssetParameters(BaseModel):
    """Base Pydantic model for explicit, editable generator parameters."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    seed: int = 0

    @classmethod
    def is_editable(cls, field_name: str) -> bool:
        if field_name == "seed" or field_name not in cls.model_fields:
            return False
        return _is_editable(cls.model_fields[field_name])

    @classmethod
    def editable_field_names(cls) -> frozenset[str]:
        return frozenset(
            name
            for name in cls.model_fields
            if name != "seed" and cls.is_editable(name)
        )

    def sweep(
        self, field: str, n_steps: int = 11
    ) -> tuple[list[float], list[Any]]:
        """Return (unit_values, native_values) for a uniform sweep of a field."""
        if field not in self.model_fields:
            raise KeyError(field)
        field_info = self.model_fields[field]
        kind = _field_kind(field_info)
        if kind == "bool":
            return [0.0, 1.0], [False, True]
        if kind == "draw_bool":
            return [0.0, 1.0], [0.0, 1.0]
        if kind == "enum":
            choices = _field_choices(field_info)
            if len(choices) < 2:
                raise ValueError(f"{field} enum needs >= 2 choices for sweep")
            unit_values = np.linspace(0.0, 1.0, len(choices)).tolist()
            return unit_values, list(choices)
        bounds = _field_bounds(field_info)
        if bounds is None:
            raise ValueError(f"{field} has no numeric bounds for sweep")
        low, high = bounds
        current = getattr(self, field)
        unit_values = np.linspace(0.0, 1.0, n_steps).tolist()

        if isinstance(current, int):
            native_values = [int(round(low + u * (high - low))) for u in unit_values]
        else:
            native_values = [low + u * (high - low) for u in unit_values]
        return unit_values, native_values

    def edit(self, field: str, unit_value: float) -> Self:
        """Return a new instance with field set from a unit-space value in [0, 1]."""
        if field not in self.model_fields:
            raise KeyError(field)
        field_info = self.model_fields[field]
        kind = _field_kind(field_info)
        if kind == "bool":
            return self.model_copy(update={field: bool(unit_value >= 0.5)})
        if kind == "draw_bool":
            return self.model_copy(update={field: float(unit_value >= 0.5)})
        if kind == "enum":
            choices = _field_choices(field_info)
            if len(choices) < 2:
                raise ValueError(f"{field} enum needs >= 2 choices for edit")
            idx = int(round(unit_value * (len(choices) - 1)))
            idx = max(0, min(idx, len(choices) - 1))
            return self.model_copy(update={field: choices[idx]})
        bounds = _field_bounds(field_info)
        if bounds is None:
            raise ValueError(f"{field} has no numeric bounds for edit")
        low, high = bounds
        current = getattr(self, field)
        new_value: float | int = low + unit_value * (high - low)
        if isinstance(current, int):
            new_value = int(round(new_value))
        return self.model_copy(update={field: new_value})


class LegacyBridgeParameters(AssetParameters):
    """Parameters model that accepts arbitrary legacy factory instance attrs."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


def legacy_init_to_parameters(
    params_cls: type[TParams],
    factory_cls: type,
    seed: int,
    coarse: bool,
    *args: Any,
    init_fn: Callable[..., None] | None = None,
    **kwargs: Any,
) -> TParams:
    """Sample parameters by running a factory's legacy ``__init__``."""
    from infinigen.core.placement.factory import AssetFactory

    inst = factory_cls.__new__(factory_cls)
    AssetFactory.__init__(inst, seed, coarse)
    inst._legacy_bridge_init = True
    try:
        with FixedSeed(seed):
            if init_fn is not None:
                init_fn(inst, seed, coarse, *args, **kwargs)
            else:
                factory_cls.__init__(inst, seed, coarse, *args, **kwargs)
    finally:
        inst._legacy_bridge_init = False
    data = {
        k: v for k, v in vars(inst).items() if k not in ("factory_seed", "coarse")
    }
    return params_cls(seed=seed, **data)


def apply_bridge_parameters(
    target: Any, params: AssetParameters, *, spawn_scope: bool = True
) -> None:
    """Copy all parameter fields onto a factory instance."""
    for key, value in params.model_dump(mode="python").items():
        if key != "seed":
            setattr(target, key, value)
    extra = getattr(params, "__pydantic_extra__", None)
    if extra:
        for key, value in extra.items():
            setattr(target, key, value)
    target._use_fixed_spawn_draws = spawn_scope


class ParameterizedAssetFactory:
    """Mixin adding sample_parameters / generate to AssetFactory subclasses."""

    parameters_model: ClassVar[type[AssetParameters]]

    def sample_parameters(
        self, seed: int | None = None, *, i: int | None = None
    ) -> AssetParameters:
        effective_seed = self.factory_seed if seed is None else seed
        asset_index = effective_seed if i is None else i
        with FixedSeed(effective_seed):
            params = self._sample_init_parameters(effective_seed)
        with FixedSeed(int_hash((effective_seed, asset_index))):
            params = self._apply_spawn_parameters(params, effective_seed, asset_index)
        params = params.model_copy(update={"seed": effective_seed})
        return params

    def _sample_init_parameters(self, seed: int) -> AssetParameters:
        raise NotImplementedError

    def _sample_spawn_parameters(
        self, params: AssetParameters, seed: int, i: int
    ) -> AssetParameters:
        return params

    def _preserve_editable_spawn_fields(
        self, params: AssetParameters, fresh: AssetParameters
    ) -> AssetParameters:
        """Keep quartet-edited values when spawn resampling would overwrite them."""
        editable = type(params).editable_field_names()
        preserved = {name: getattr(params, name) for name in editable}
        return fresh.model_copy(update=preserved)

    def _apply_spawn_parameters(
        self, params: AssetParameters, seed: int, i: int
    ) -> AssetParameters:
        fresh = self._sample_spawn_parameters(params, seed, i)
        return self._preserve_editable_spawn_fields(params, fresh)

    def apply_parameters(
        self, params: AssetParameters, *, spawn_scope: bool = True
    ) -> None:
        raise NotImplementedError

    def init_legacy_parameters(self) -> None:
        self._use_fixed_spawn_draws = False
        with FixedSeed(self.factory_seed):
            params = self._sample_init_parameters(self.factory_seed)
            self.apply_parameters(params, spawn_scope=False)
        self._run_post_init()

    def _run_post_init(self) -> None:
        post_init = getattr(self, "post_init", None)
        if callable(post_init):
            with FixedSeed(self.factory_seed):
                post_init()

    def _needs_placeholder(self) -> bool:
        """Return True when create_asset requires a placeholder positional arg."""
        param = inspect.signature(self.create_asset).parameters.get("placeholder")
        return param is not None and param.default is inspect.Parameter.empty

    def generate(
        self,
        params: AssetParameters,
        i: int | None = None,
        distance: float | None = None,
        vis_distance: float = 0,
        **kwargs: Any,
    ) -> bpy.types.Object:
        from . import detail

        asset_index = params.seed if i is None else i
        with FixedSeed(int_hash((self.factory_seed, asset_index))):
            params = self._apply_spawn_parameters(params, params.seed, asset_index)
        self.apply_parameters(params, spawn_scope=True)
        if distance is None:
            distance = detail.scatter_res_distance()
        if self._needs_placeholder():
            return self.spawn_asset(
                i=asset_index,
                distance=distance,
                vis_distance=vis_distance,
                **kwargs,
            )
        with FixedSeed(int_hash((self.factory_seed, asset_index))):
            spawn_params = self.asset_parameters(distance, vis_distance)
            spawn_params.update(kwargs)
            return self.create_asset(i=asset_index, **spawn_params)
