# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from __future__ import annotations

from typing import Annotated, ClassVar

import bpy
import numpy as np
from numpy.random import uniform
from pydantic import Field

from infinigen.assets.objects.tableware.base import (
    TablewareFactory,
    apply_tableware_from_draws,
    sample_tableware_base,
)
from infinigen.assets.utils.decorate import subsurf, write_co
from infinigen.assets.utils.object import join_objects, new_grid
from infinigen.core.placement.parameters import AssetParameters, ParameterizedAssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform


class ChopsticksParameters(AssetParameters):
    y_shrink: Annotated[float, Field(ge=0.2, le=0.8, json_schema_extra={"editable": False})]
    is_square: Annotated[
        bool, Field(json_schema_extra={"editable": False, "kind": "bool"})
    ]
    has_guard: Annotated[
        bool, Field(json_schema_extra={"editable": True, "kind": "bool"})
    ]
    lower_thresh: Annotated[float, Field(ge=0.5, le=0.8, json_schema_extra={"editable": False})]
    scale: Annotated[float, Field(ge=0.2, le=0.4, json_schema_extra={"editable": True})]
    scratch_draw: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            json_schema_extra={"editable": False, "kind": "draw_bool"},
        ),
    ]
    thickness: float = Field(default=0.01, json_schema_extra={"editable": False})
    edge_wear_draw: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            json_schema_extra={"editable": False, "kind": "draw_bool"},
        ),
    ]
    parallel_style: Annotated[
        bool, Field(json_schema_extra={"editable": True, "kind": "bool"})
    ] = True
    is_parallel: Annotated[
        bool, Field(json_schema_extra={"editable": True, "kind": "bool"})
    ] = True
    parallel_distance: Annotated[
        float, Field(ge=0.01, le=0.04, json_schema_extra={"editable": True})
    ] = 0.01
    parallel_rot_a: Annotated[
        float, Field(ge=0.0, le=0.392699, json_schema_extra={"editable": True})
    ] = 0.0
    parallel_rot_b: Annotated[
        float, Field(ge=0.0, le=0.392699, json_schema_extra={"editable": True})
    ] = 0.0
    crossed_loc_x: Annotated[
        float, Field(ge=-0.1, le=0.2, json_schema_extra={"editable": True})
    ] = 0.0
    crossed_loc_y: Annotated[
        float, Field(ge=-0.2, le=0.2, json_schema_extra={"editable": True})
    ] = 0.0
    crossed_rot: Annotated[
        float, Field(ge=0.392699, le=0.785398, json_schema_extra={"editable": True})
    ] = 0.392699


class ChopsticksFactory(ParameterizedAssetFactory, TablewareFactory):
    parameters_model: ClassVar[type[AssetParameters]] = ChopsticksParameters

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        self.pre_level = 2
        self.init_legacy_parameters()

    def _layout_branch_updates(
        self,
        *,
        y_length: float,
        is_parallel: bool,
        parallel_style: bool | None = None,
    ) -> dict[str, bool | float]:
        if is_parallel:
            return {
                "is_parallel": True,
                "parallel_style": (
                    parallel_style
                    if parallel_style is not None
                    else bool(uniform(0, 1) < 0.5)
                ),
                "parallel_distance": log_uniform(y_length, 0.04),
                "parallel_rot_a": uniform(0, np.pi / 8),
                "parallel_rot_b": uniform(0, np.pi / 8),
            }
        crossed_loc_y = uniform(-0.2, 0.2)
        sign = np.sign(crossed_loc_y)
        return {
            "is_parallel": False,
            "crossed_loc_x": uniform(-0.1, 0.2),
            "crossed_loc_y": crossed_loc_y,
            "crossed_rot": log_uniform(np.pi / 8, np.pi / 4) * sign,
        }

    def _sample_init_parameters(self, seed: int) -> ChopsticksParameters:
        base = sample_tableware_base(seed)
        with FixedSeed(seed):
            self._y_length = uniform(0.01, 0.02)
            is_parallel = bool(uniform(0, 1) < 0.6)
            layout = self._layout_branch_updates(
                y_length=self._y_length, is_parallel=is_parallel
            )
        return ChopsticksParameters(
            seed=seed,
            y_shrink=log_uniform(0.2, 0.8),
            is_square=bool(uniform(0, 1) < 0.5),
            has_guard=bool(uniform(0, 1) < 0.4),
            lower_thresh=base["lower_thresh"],
            scale=log_uniform(0.2, 0.4),
            scratch_draw=base["scratch_draw"],
            edge_wear_draw=base["edge_wear_draw"],
            **layout,
        )

    def _sample_spawn_parameters(
        self, params: ChopsticksParameters, seed: int, i: int
    ) -> ChopsticksParameters:
        editable = params.editable_field_names()
        is_parallel = (
            params.is_parallel
            if "is_parallel" in editable
            else bool(uniform(0, 1) < 0.6)
        )
        parallel_style = (
            params.parallel_style if "parallel_style" in editable else None
        )
        layout = self._layout_branch_updates(
            y_length=self._y_length,
            is_parallel=is_parallel,
            parallel_style=parallel_style,
        )
        return params.model_copy(update=layout)

    def apply_parameters(
        self, params: ChopsticksParameters, *, spawn_scope: bool = True
    ) -> None:
        apply_tableware_from_draws(
            self,
            seed=params.seed,
            lower_thresh=params.lower_thresh,
            scale=params.scale,
            scratch_draw=params.scratch_draw,
            edge_wear_draw=params.edge_wear_draw,
            has_guard=params.has_guard,
            guard_depth=params.thickness * 2 if params.has_guard else 0.0,
        )
        self.thickness = 0.01
        # NOTE: y_length resampled in spawn path overwrote edits; sampled on self from seed, excluded from quartet sampling.
        with FixedSeed(params.seed):
            self.y_length = uniform(0.01, 0.02)
        self.y_shrink = params.y_shrink
        self.is_square = params.is_square
        self.has_guard = params.has_guard
        # NOTE: x_guard does not elicit a clear visual change in exported geometry; excluded from quartet sampling.
        with FixedSeed(params.seed):
            self.x_guard = uniform(0.4, 0.9)
        self.guard_depth = params.thickness * 2 if params.has_guard else 0.0
        self._use_fixed_spawn_draws = spawn_scope
        if spawn_scope:
            self._is_parallel = params.is_parallel
            self._parallel_style = params.parallel_style
            self._parallel_distance = params.parallel_distance
            self._parallel_rot_a = params.parallel_rot_a
            self._parallel_rot_b = params.parallel_rot_b
            self._crossed_loc_x = params.crossed_loc_x
            self._crossed_loc_y = params.crossed_loc_y
            self._crossed_rot = params.crossed_rot

    def create_asset(self, **params) -> bpy.types.Object:
        obj = self.make_single()
        if self._use_fixed_spawn_draws:
            is_parallel = self._is_parallel
        else:
            is_parallel = uniform(0, 1) < 0.6
        if is_parallel:
            obj = self.make_parallel(obj)
        else:
            obj = self.make_crossed(obj)
        return obj

    def make_parallel(self, obj):
        if self._use_fixed_spawn_draws:
            distance = self._parallel_distance
            parallel_style = self._parallel_style
            rot_a = self._parallel_rot_a
            rot_b = self._parallel_rot_b
        else:
            distance = log_uniform(self.y_length, 0.04)
            parallel_style = uniform(0, 1) < 0.5
            rot_a = uniform(0, np.pi / 8)
            rot_b = uniform(0, np.pi / 8)
        if parallel_style:
            other = deep_clone_obj(obj)
            obj.location[1] = distance
            obj.rotation_euler[-1] = rot_a
            other.location[1] = -distance
            other.rotation_euler[-1] = -rot_b
        else:
            obj.location[0] = -1
            butil.apply_transform(obj, loc=True)
            other = deep_clone_obj(obj)
            obj.location[1] = distance
            obj.rotation_euler[-1] = -rot_b
            other.location[1] = -distance
            other.rotation_euler[-1] = rot_a
        return join_objects([obj, other])

    def make_crossed(self, obj):
        other = deep_clone_obj(obj)
        if self._use_fixed_spawn_draws:
            other.location = (
                self._crossed_loc_x,
                self._crossed_loc_y,
                self.y_length,
            )
            other.rotation_euler[-1] = -self._crossed_rot
        else:
            other.location = uniform(-0.1, 0.2), uniform(-0.2, 0.2), self.y_length
            sign = np.sign(other.location[1])
            other.rotation_euler[-1] = -sign * log_uniform(np.pi / 8, np.pi / 4)
        return join_objects([obj, other])

    def make_single(self):
        n = int(1 / self.y_length)
        obj = new_grid(x_subdivisions=n - 1, y_subdivisions=1)
        butil.modify_mesh(obj, "SOLIDIFY", thickness=self.y_length * 2)
        l = np.linspace(self.y_shrink, 1, n) * self.y_length
        x = np.concatenate([np.linspace(0, 1, n)] * 4)
        y = np.concatenate([-l, l, -l, l])
        z = np.concatenate([l, l, -l, -l])
        write_co(obj, np.stack([x, y, z], -1))
        subsurf(obj, 2, self.is_square)
        self.add_guard(obj, lambda nw, x: nw.compare("GREATER_THAN", x, self.x_guard))
        obj.scale = [self.scale] * 3
        butil.apply_transform(obj)
        return obj
