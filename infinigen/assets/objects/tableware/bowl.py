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
from infinigen.assets.utils.decorate import set_shade_smooth, subsurf
from infinigen.assets.utils.draw import spin
from infinigen.assets.utils.object import new_bbox
from infinigen.core.placement.parameters import AssetParameters, ParameterizedAssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform


class BowlParameters(AssetParameters):
    thickness_ratio: Annotated[
        float, Field(ge=0.01, le=0.03, json_schema_extra={"editable": True})
    ]
    x_mid: Annotated[float, Field(ge=0.8, le=0.95, json_schema_extra={"editable": True})]
    z_length: Annotated[float, Field(ge=0.4, le=0.8, json_schema_extra={"editable": True})]
    has_inside: Annotated[
        bool, Field(json_schema_extra={"editable": True, "kind": "bool"})
    ] = True
    bevel_segments: Annotated[int, Field(ge=2, le=4, json_schema_extra={"editable": False})] = (
        2
    )


class BowlFactory(ParameterizedAssetFactory, TablewareFactory):
    allow_transparent = True
    parameters_model: ClassVar[type[AssetParameters]] = BowlParameters

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        self.x_end = 0.5
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> BowlParameters:
        return BowlParameters(
            seed=seed,
            thickness_ratio=uniform(0.01, 0.03),
            x_mid=uniform(0.8, 0.95),
            z_length=log_uniform(0.4, 0.8),
            has_inside=uniform() < 0.5,
        )

    def _sample_spawn_parameters(
        self, params: BowlParameters, seed: int, i: int
    ) -> BowlParameters:
        return params.model_copy(update={"bevel_segments": int(np.random.randint(2, 5))})

    def apply_parameters(
        self, params: BowlParameters, *, spawn_scope: bool = True
    ) -> None:
        # NOTE: scale sampled on self from seed; excluded from quartet sampling (uniform scale normalized away in point clouds).
        with FixedSeed(params.seed):
            self.scale = log_uniform(0.15, 0.4)
            base = sample_tableware_base(params.seed)
            self._lower_thresh = base["lower_thresh"]
        apply_tableware_from_draws(
            self,
            seed=params.seed,
            lower_thresh=self._lower_thresh,
            scale=self.scale,
        )
        self.has_inside = params.has_inside
        # NOTE: x_bottom and z_bottom do not elicit a clear visual change in exported geometry; excluded from quartet sampling.
        with FixedSeed(params.seed):
            self.x_bottom = uniform(0.2, 0.3) * self.x_end
            self.z_bottom = log_uniform(0.02, 0.05)
        self.x_mid = params.x_mid * self.x_end
        self.z_length = params.z_length
        self.thickness = params.thickness_ratio * self.z_length * 2
        self._use_fixed_spawn_draws = spawn_scope
        if spawn_scope:
            self._bevel_segments = params.bevel_segments

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        radius = self.x_end * self.scale
        return new_bbox(-radius, radius, -radius, radius, 0, self.z_length * self.scale)

    def create_asset(self, **params) -> bpy.types.Object:
        bevel_segments = (
            self._bevel_segments
            if self._use_fixed_spawn_draws
            else np.random.randint(2, 5)
        )
        x_anchors = (
            0,
            self.x_bottom,
            self.x_bottom + 1e-3,
            self.x_bottom,
            self.x_mid,
            self.x_end,
        )
        z_anchors = 0, 0, 0, self.z_bottom, self.z_length / 2, self.z_length
        anchors = np.array(x_anchors) * self.scale, 0, np.array(z_anchors) * self.scale
        obj = spin(anchors, [2, 3])
        self.solidify_with_inside(obj, self.thickness)
        butil.modify_mesh(obj, "BEVEL", width=self.thickness / 2, segments=bevel_segments)
        subsurf(obj, 1)
        set_shade_smooth(obj)
        return obj
