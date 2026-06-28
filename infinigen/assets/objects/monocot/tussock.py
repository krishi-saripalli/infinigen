# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei

from __future__ import annotations

from typing import Annotated, ClassVar

import numpy as np
from numpy.random import uniform
from pydantic import Field

from infinigen.assets.objects.monocot.growth import MonocotGrowthFactory
from infinigen.assets.utils.draw import leaf
from infinigen.core.placement.parameters import AssetParameters, ParameterizedAssetFactory
from infinigen.core.surface import shaderfunc_to_material
from infinigen.core.tagging import tag_object
from infinigen.core.util.color import hsv2rgba
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform


class TussockMonocotParameters(AssetParameters):
    stem_offset: Annotated[float, Field(ge=0.0, le=0.2, json_schema_extra={"editable": False})]
    min_y_angle: Annotated[
        float, Field(ge=0.628319, le=0.785398, json_schema_extra={"editable": False})
    ]
    count: Annotated[float, Field(ge=512.0, le=1024.0, json_schema_extra={"editable": False})]
    scale_curve_high: Annotated[
        float, Field(ge=0.6, le=1.0, json_schema_extra={"editable": False})
    ]


class TussockMonocotFactory(ParameterizedAssetFactory, MonocotGrowthFactory):
    parameters_model: ClassVar[type[AssetParameters]] = TussockMonocotParameters

    def __init__(self, factory_seed, coarse=False):
        super(TussockMonocotFactory, self).__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_material(self, seed: int) -> None:
        # NOTE: base_hue and base_hue_draw are sampled on self in apply_parameters; excluded from quartet sampling (material-only, not exported geometry).
        with FixedSeed(seed):
            base_hue_draw = uniform(0, 1)
            if base_hue_draw < 0.5:
                base_hue = uniform(0.1, 0.15)
            else:
                base_hue = uniform(0.25, 0.35)
            bright_color = hsv2rgba(base_hue, uniform(0.6, 0.8), log_uniform(0.05, 0.1))
            dark_color = hsv2rgba(
                (base_hue + uniform(-0.03, 0.03)) % 1,
                uniform(0.8, 1.0),
                log_uniform(0.05, 0.2),
            )
        self.base_hue = base_hue
        self.material = shaderfunc_to_material(
            self.shader_monocot, dark_color, bright_color, self.use_distance
        )

    def _sample_init_parameters(self, seed: int) -> TussockMonocotParameters:
        self._sample_material(seed)
        return TussockMonocotParameters(
            seed=seed,
            stem_offset=uniform(0.0, 0.2),
            min_y_angle=uniform(np.pi * 0.2, np.pi * 0.25),
            count=log_uniform(512, 1024),
            scale_curve_high=uniform(0.6, 1.0),
        )

    def apply_parameters(
        self, params: TussockMonocotParameters, *, spawn_scope: bool = True
    ) -> None:
        self._sample_material(params.seed)
        # NOTE: leaf_prob, z_drag, z_scale, angle, and scale_curve_low do not elicit a reliable visual change in exported geometry; sampled on self from seed, excluded from quartet sampling.
        with FixedSeed(params.seed):
            self.leaf_prob = uniform(0.8, 0.9)
            self.z_drag = uniform(0.1, 0.2)
            self.z_scale = uniform(1.0, 1.2)
            self.angle = uniform(np.pi / 20, np.pi / 18)
            self.scale_curve_low = uniform(0.6, 1.0)
        self.count = int(params.count)
        self.perturb = 0.05
        # NOTE: stem_offset, min_y_angle, and scale_curve_high have intermittent effect across leaf instances; excluded from quartet sampling.
        self.min_y_angle = params.min_y_angle
        self.max_y_angle = np.pi / 2
        self.leaf_range = 0, 1
        self.stem_offset = params.stem_offset
        self.scale_curve = [
            (0, self.scale_curve_low),
            (1, params.scale_curve_high),
        ]
        self.radius = 0.01
        self.bend_angle = np.pi / 4
        self.twist_angle = np.pi / 6
        self.align_factor = 0
        self.align_direction = 1, 0, 0
        self._use_fixed_spawn_draws = spawn_scope
        self._cache_decor_state(params.seed)

    @staticmethod
    def build_base_hue():
        if uniform(0, 1) < 0.5:
            return uniform(0.1, 0.15)
        else:
            return uniform(0.25, 0.35)

    def build_leaf(self, face_size):
        x_anchors = np.array([0, uniform(0.3, 0.7), 1.0])
        y_anchors = np.array([0, 0.01, 0])
        obj = leaf(x_anchors, y_anchors, face_size=face_size)
        self.decorate_leaf(obj)
        tag_object(obj, "tussock")
        return obj
