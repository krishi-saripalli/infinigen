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
from infinigen.assets.utils.decorate import subsurf
from infinigen.assets.utils.draw import spin
from infinigen.core.placement.parameters import AssetParameters, ParameterizedAssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform


class PlateParameters(AssetParameters):
    thickness_ratio: Annotated[
        float, Field(ge=0.01, le=0.03, json_schema_extra={"editable": True})
    ]
    lower_thresh: Annotated[float, Field(ge=0.5, le=0.8, json_schema_extra={"editable": False})]
    x_mid: Annotated[float, Field(ge=0.3, le=1.0, json_schema_extra={"editable": True})]
    z_length: Annotated[float, Field(ge=0.05, le=0.2, json_schema_extra={"editable": True})]
    z_mid_ratio: Annotated[
        float, Field(ge=0.3, le=0.8, json_schema_extra={"editable": False})
    ]
    has_inside: Annotated[
        bool, Field(json_schema_extra={"editable": False, "kind": "bool"})
    ]
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


class PlateFactory(ParameterizedAssetFactory, TablewareFactory):
    allow_transparent = True
    parameters_model: ClassVar[type[AssetParameters]] = PlateParameters

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        self.x_end = 0.5
        self.pre_level = 1
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> PlateParameters:
        base = sample_tableware_base(seed)
        z_length = log_uniform(0.05, 0.2)
        return PlateParameters(
            seed=seed,
            thickness_ratio=uniform(0.01, 0.03),
            lower_thresh=base["lower_thresh"],
            x_mid=uniform(0.3, 1.0),
            z_length=z_length,
            z_mid_ratio=uniform(0.3, 0.8),
            has_inside=bool(uniform() < 0.2),
            scratch_draw=base["scratch_draw"],
            edge_wear_draw=base["edge_wear_draw"],
        )

    def apply_parameters(
        self, params: PlateParameters, *, spawn_scope: bool = True
    ) -> None:
        # NOTE: scale sampled on self from seed; excluded from quartet sampling (uniform scale normalized away in point clouds).
        with FixedSeed(params.seed):
            self.scale = log_uniform(0.2, 0.4)
        apply_tableware_from_draws(
            self,
            seed=params.seed,
            lower_thresh=params.lower_thresh,
            scale=self.scale,
            scratch_draw=params.scratch_draw,
            edge_wear_draw=params.edge_wear_draw,
        )
        self.has_inside = params.has_inside
        self.x_mid = params.x_mid * self.x_end
        self.z_length = params.z_length
        self.z_mid = params.z_mid_ratio * params.z_length
        self.thickness = params.thickness_ratio * self.z_length * 4
        self._use_fixed_spawn_draws = spawn_scope

    def create_asset(self, **params) -> bpy.types.Object:
        x_anchors = 0, self.x_mid, self.x_mid, self.x_end
        z_anchors = 0, 0, self.z_mid, self.z_length
        anchors = np.array(x_anchors) * self.scale, 0, np.array(z_anchors) * self.scale
        obj = spin(anchors, [1, 2])
        butil.modify_mesh(
            obj, "SUBSURF", render_levels=self.pre_level, levels=self.pre_level
        )
        self.solidify_with_inside(obj, self.thickness)
        subsurf(obj, 1)
        return obj
