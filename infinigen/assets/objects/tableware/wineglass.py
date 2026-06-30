# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from __future__ import annotations

from typing import Annotated, ClassVar

import bpy
import numpy as np
from numpy.random import uniform
from pydantic import Field

from infinigen.assets.composition import material_assignments
from infinigen.assets.objects.tableware.base import TablewareFactory, sample_tableware_base
from infinigen.assets.utils.draw import spin
from infinigen.core.placement.parameters import AssetParameters, ParameterizedAssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform, weighted_sample


class WineglassParameters(AssetParameters):
    z_length: Annotated[float, Field(ge=0.6, le=2.0, json_schema_extra={"editable": True})]
    z_cup: Annotated[float, Field(ge=0.3, le=0.6, json_schema_extra={"editable": True})]
    z_mid: Annotated[float, Field(ge=0.3, le=0.5, json_schema_extra={"editable": True})]
    x_top: Annotated[float, Field(ge=1.0, le=1.4, json_schema_extra={"editable": True})]
    x_mid: Annotated[float, Field(ge=0.9, le=1.2, json_schema_extra={"editable": True})]
    thickness: Annotated[float, Field(ge=0.01, le=0.03, json_schema_extra={"editable": False})]
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

class WineglassFactory(ParameterizedAssetFactory, TablewareFactory):
    parameters_model: ClassVar[type[AssetParameters]] = WineglassParameters

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        self.x_end = 0.25
        self.has_guard = False
        self.init_legacy_parameters()

    def _resolve_tableware(self, params: WineglassParameters) -> None:
        base = sample_tableware_base(params.seed)
        scratch_prob, edge_wear_prob = base["scratch_prob"], base["edge_wear_prob"]
        self.surface = weighted_sample(material_assignments.glasses)()()
        self.inside_surface = base["inside_surface"]
        self.guard_surface = base["guard_surface"]
        self.scratch = (
            None
            if params.scratch_draw > scratch_prob
            else base["scratch_fn"]()
        )
        self.edge_wear = (
            None
            if params.edge_wear_draw > edge_wear_prob
            else base["edge_wear_fn"]()
        )
        self.has_guard = False
        self.guard_depth = base["guard_depth"]
        self.metal_color = base["metal_color"]
        self.lower_thresh = base["lower_thresh"]
        self.thickness = params.thickness

    def _sample_wineglass_scale(self, seed: int) -> float:
        # NOTE: scale sampled on self from seed; excluded from quartet sampling (uniform scale normalized away in point clouds).
        with FixedSeed(seed):
            return log_uniform(0.1, 0.3)

    def _sample_init_parameters(self, seed: int) -> WineglassParameters:
        base = sample_tableware_base(seed)
        return WineglassParameters(
            seed=seed,
            z_length=log_uniform(0.6, 2.0),
            z_cup=uniform(0.3, 0.6),
            z_mid=uniform(0.3, 0.5),
            x_top=log_uniform(1, 1.4),
            x_mid=log_uniform(0.9, 1.2),
            thickness=uniform(0.01, 0.03),
            scratch_draw=base["scratch_draw"],
            edge_wear_draw=base["edge_wear_draw"],
        )

    def apply_parameters(
        self, params: WineglassParameters, *, spawn_scope: bool = True
    ) -> None:
        # NOTE: lower_thresh does not elicit a reliable visual change in exported geometry; sampled on self from seed, excluded from quartet sampling.
        self._resolve_tableware(params)
        self.scale = self._sample_wineglass_scale(params.seed)
        self.z_length = params.z_length
        z_cup_abs = params.z_cup * params.z_length
        self.z_cup = z_cup_abs
        # NOTE: z_mid and x_mid anchor positions vary with z_cup/z_length ratio branches; thickness SOLIDIFY delta is sub-pixel after normalization; excluded from quartet sampling.
        self.z_mid = z_cup_abs + params.z_mid * (params.z_length - z_cup_abs)
        # NOTE: x_neck does not elicit a clear visual change in exported geometry; excluded from quartet sampling.
        with FixedSeed(params.seed):
            self.x_neck = log_uniform(0.01, 0.02)
        self.x_top = self.x_end * params.x_top
        self.x_mid = self.x_top * params.x_mid
        with FixedSeed(params.seed):
            self._z_bottom = log_uniform(0.01, 0.05)
        self._use_fixed_spawn_draws = spawn_scope

    def create_asset(self, **params) -> bpy.types.Object:
        z_bottom = self.z_length * (
            self._z_bottom if self._use_fixed_spawn_draws else log_uniform(0.01, 0.05)
        )
        x_anchors = (
            self.x_end,
            self.x_end / 2,
            self.x_neck,
            self.x_neck,
            self.x_mid,
            self.x_top,
        )
        z_anchors = 0, z_bottom / 2, z_bottom, self.z_cup, self.z_mid, self.z_length
        anchors = x_anchors, np.zeros_like(x_anchors), z_anchors
        obj = spin(anchors, [0, 1, 2, 3])
        butil.modify_mesh(obj, "SOLIDIFY", thickness=self.thickness)
        obj.scale = [self.scale] * 3
        butil.apply_transform(obj)

        with butil.SelectObjects(obj):
            bpy.ops.object.shade_smooth()

        return obj
