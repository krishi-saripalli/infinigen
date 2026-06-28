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
from infinigen.assets.utils.object import new_grid
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import (
    AssetParameters,
    ParameterizedAssetFactory,
)
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform


class SpoonParameters(AssetParameters):
    x_length: Annotated[float, Field(ge=0.2, le=0.8, json_schema_extra={"editable": True})]
    y_length: Annotated[float, Field(ge=0.06, le=0.12, json_schema_extra={"editable": False})]
    thickness: Annotated[float, Field(ge=0.008, le=0.015, json_schema_extra={"editable": False})]
    guard_depth_mult: Annotated[
        float, Field(ge=0.2, le=1.0, json_schema_extra={"editable": False})
    ]
    has_guard: Annotated[
        bool, Field(json_schema_extra={"editable": False, "kind": "bool"})
    ] = True
    scratch_draw: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            json_schema_extra={"editable": False, "kind": "draw_bool"},
        ),
    ]
    edge_wear_draw: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            json_schema_extra={"editable": False, "kind": "draw_bool"},
        ),
    ]
    x_anchor_head: Annotated[
        float, Field(ge=0.07, le=0.25, json_schema_extra={"editable": False})
    ] = 0.15
    x_anchor_tail_mult: Annotated[
        float, Field(ge=1.2, le=1.4, json_schema_extra={"editable": False})
    ] = 1.3
    y_anchor_0_mult: Annotated[
        float, Field(ge=0.1, le=0.8, json_schema_extra={"editable": False})
    ] = 0.45
    y_anchor_5: Annotated[
        float, Field(ge=0.02, le=0.05, json_schema_extra={"editable": False})
    ] = 0.035
    z_anchor_mid: Annotated[
        float, Field(ge=-0.02, le=0.04, json_schema_extra={"editable": False})
    ] = 0.01


class SpoonFactory(ParameterizedAssetFactory, TablewareFactory):
    parameters_model: ClassVar[type[AssetParameters]] = SpoonParameters
    x_end = 0.15
    is_fragile = True

    def _sample_spawn_anchor_updates(self) -> dict[str, float]:
        return {
            "x_anchor_head": log_uniform(0.07, 0.25),
            "x_anchor_tail_mult": log_uniform(1.2, 1.4),
            "y_anchor_0_mult": log_uniform(0.1, 0.8),
            "y_anchor_5": log_uniform(0.02, 0.05),
            "z_anchor_mid": uniform(-0.02, 0.04),
        }

    def __init__(self, factory_seed, coarse=False):
        AssetFactory.__init__(self, factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> SpoonParameters:
        base = sample_tableware_base(seed)
        thickness = log_uniform(0.008, 0.015)
        return SpoonParameters(
            seed=seed,
            x_length=log_uniform(0.2, 0.8),
            y_length=log_uniform(0.06, 0.12),
            thickness=thickness,
            guard_depth_mult=log_uniform(0.2, 1.0),
            has_guard=True,
            scratch_draw=base["scratch_draw"],
            edge_wear_draw=base["edge_wear_draw"],
            **self._sample_spawn_anchor_updates(),
        )

    def _sample_spawn_parameters(
        self, params: SpoonParameters, seed: int, i: int
    ) -> SpoonParameters:
        return params.model_copy(update=self._sample_spawn_anchor_updates())

    def apply_parameters(
        self, params: SpoonParameters, *, spawn_scope: bool = True
    ) -> None:
        with FixedSeed(params.seed):
            self.scale = log_uniform(0.15, 0.25)
            base = sample_tableware_base(params.seed)
            self._lower_thresh = base["lower_thresh"]
            self._y_anchor_1_mult = log_uniform(1.0, 1.2)
            self._y_anchor_2_mult = log_uniform(0.6, 1.0)
            self._y_anchor_3_mult = log_uniform(0.2, 0.4)
            self._y_anchor_4 = log_uniform(0.01, 0.02)
            self._y_anchor_6 = log_uniform(0.01, 0.02)
            self._z_anchor_tail = uniform(-0.02, 0.0)
        apply_tableware_from_draws(
            self,
            seed=params.seed,
            lower_thresh=self._lower_thresh,
            scale=self.scale,
            scratch_draw=params.scratch_draw,
            edge_wear_draw=params.edge_wear_draw,
            guard_depth=params.guard_depth_mult * params.thickness,
        )
        self.x_length = params.x_length
        self.y_length = params.y_length
        with FixedSeed(params.seed):
            self.z_depth = log_uniform(0.08, 0.25)
            self.z_offset = uniform(0.0, 0.05)
        self.thickness = params.thickness
        self.has_guard = params.has_guard
        with FixedSeed(params.seed):
            self.guard_type = "round" if uniform(0, 1) < 0.6 else "double"
        self.guard_depth = params.guard_depth_mult * params.thickness
        self._x_anchor_head = params.x_anchor_head
        self._x_anchor_tail_mult = params.x_anchor_tail_mult
        self._y_anchor_0_mult = params.y_anchor_0_mult
        self._y_anchor_5 = params.y_anchor_5
        self._z_anchor_mid = params.z_anchor_mid
        self._use_fixed_spawn_draws = spawn_scope

    def create_asset(self, **params) -> bpy.types.Object:
        if self._use_fixed_spawn_draws:
            x_head = self._x_anchor_head
            x_tail_mult = self._x_anchor_tail_mult
            y0 = self.y_length * self._y_anchor_0_mult
            y1 = self.y_length * self._y_anchor_1_mult
            y2 = self.y_length * self._y_anchor_2_mult
            y3 = self.y_length * self._y_anchor_3_mult
            y4 = self._y_anchor_4
            y5 = self._y_anchor_5
            y6 = self._y_anchor_6
            z_mid = self._z_anchor_mid
            z_tail = self._z_anchor_tail
        else:
            x_head = log_uniform(0.07, 0.25)
            x_tail_mult = log_uniform(1.2, 1.4)
            y0 = self.y_length * log_uniform(0.1, 0.8)
            y1 = self.y_length * log_uniform(1.0, 1.2)
            y2 = self.y_length * log_uniform(0.6, 1.0)
            y3 = self.y_length * log_uniform(0.2, 0.4)
            y4 = log_uniform(0.01, 0.02)
            y5 = log_uniform(0.02, 0.05)
            y6 = log_uniform(0.01, 0.02)
            z_mid = uniform(-0.02, 0.04)
            z_tail = uniform(-0.02, 0.0)
        x_anchors = np.array(
            [
                x_head,
                0,
                -0.08,
                -0.12,
                -self.x_end,
                -self.x_end - self.x_length,
                -self.x_end - self.x_length * x_tail_mult,
            ]
        )
        y_anchors = np.array([y0, y1, y2, y3, y4, y5, y6])
        z_anchors = np.array(
            [
                0,
                0,
                0,
                0,
                self.z_offset,
                self.z_offset + z_mid,
                self.z_offset + z_tail,
            ]
        )
        obj = new_grid(x_subdivisions=len(x_anchors) - 1, y_subdivisions=2)
        x = np.concatenate([x_anchors] * 3)
        y = np.concatenate([y_anchors, np.zeros_like(y_anchors), -y_anchors])
        z = np.concatenate([z_anchors] * 3)
        x[len(x_anchors)] += 0.02
        z[len(x_anchors) + 1] = -self.z_depth
        write_co(obj, np.stack([x, y, z], -1))
        butil.modify_mesh(obj, "SOLIDIFY", thickness=self.thickness)
        subsurf(obj, 1)

        def selection(nw, x):
            return nw.compare("LESS_THAN", x, -self.x_end)

        if self.guard_type == "double":
            selection = self.make_double_sided(selection)
        self.add_guard(obj, selection)
        subsurf(obj, 2)
        obj.scale = [self.scale] * 3
        butil.apply_transform(obj)
        return obj
