# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei

from __future__ import annotations

from typing import Annotated, ClassVar

import bpy
import numpy as np
from numpy.random import uniform
from pydantic import Field

import infinigen.core.util.blender as butil
from infinigen.assets.objects.creatures.util.animation.driver_repeated import (
    repeated_driver,
)
from infinigen.assets.objects.monocot.growth import MonocotGrowthFactory
from infinigen.assets.utils.draw import bezier_curve, leaf
from infinigen.assets.utils.misc import assign_material
from infinigen.assets.utils.object import join_objects, origin2leftmost
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.placement.detail import remesh_with_attrs
from infinigen.core.placement.parameters import (
    AssetParameters,
    ParameterizedAssetFactory,
)
from infinigen.core.surface import shaderfunc_to_material
from infinigen.core.util.color import hsv2rgba
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform


class KelpMonocotParameters(AssetParameters):
    angle: Annotated[float, Field(ge=0.523599, le=0.785398, json_schema_extra={"editable": True})]
    z_drag: Annotated[float, Field(ge=0.0, le=0.2, json_schema_extra={"editable": True})]
    min_y_angle: Annotated[
        float, Field(ge=0.0, le=0.314159, json_schema_extra={"editable": True})
    ]
    bend_angle: Annotated[
        float, Field(ge=0.0, le=0.523599, json_schema_extra={"editable": True})
    ]
    twist_angle: Annotated[
        float, Field(ge=0.0, le=0.523599, json_schema_extra={"editable": True})
    ]
    leaf_prob: Annotated[float, Field(ge=0.6, le=0.7, json_schema_extra={"editable": True})]
    flow_angle: Annotated[
        float, Field(ge=0.0, le=6.283185, json_schema_extra={"editable": True})
    ]
    align_direction_z: Annotated[
        float, Field(ge=-0.2, le=0.2, json_schema_extra={"editable": True})
    ]
    anim_period: Annotated[
        float, Field(ge=100.0, le=200.0, json_schema_extra={"editable": True})
    ]
    anim_offset: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    anim_seed: Annotated[int, Field(ge=0, le=100000, json_schema_extra={"editable": True})]
    z_scale: Annotated[float, Field(ge=1.0, le=1.2, json_schema_extra={"editable": True})]


class KelpMonocotFactory(ParameterizedAssetFactory, MonocotGrowthFactory):
    parameters_model: ClassVar[type[AssetParameters]] = KelpMonocotParameters
    max_leaf_length = 1.2
    align_angle = uniform(np.pi / 24, np.pi / 12)

    def __init__(self, factory_seed, coarse=False):
        super(KelpMonocotFactory, self).__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_material(self, seed: int) -> None:
        with FixedSeed(seed):
            base_hue = uniform(0.05, 0.25)
            bright_color = hsv2rgba(base_hue, uniform(0.6, 0.8), log_uniform(0.05, 0.1))
            dark_color = hsv2rgba(
                (base_hue + uniform(-0.03, 0.03)) % 1,
                uniform(0.8, 1.0),
                log_uniform(0.05, 0.2),
            )
        self.material = shaderfunc_to_material(
            self.shader_monocot, dark_color, bright_color, self.use_distance
        )

    def _sample_init_parameters(self, seed: int) -> KelpMonocotParameters:
        self._sample_material(seed)
        z_scale = uniform(1.0, 1.2)
        return KelpMonocotParameters(
            seed=seed,
            angle=uniform(np.pi / 6, np.pi / 4),
            z_drag=uniform(0.0, 0.2),
            min_y_angle=uniform(0, np.pi * 0.1),
            bend_angle=uniform(0, np.pi / 6),
            twist_angle=uniform(0, np.pi / 6),
            leaf_prob=uniform(0.6, 0.7),
            flow_angle=uniform(0, np.pi * 2),
            align_direction_z=uniform(-0.2, 0.2),
            anim_period=log_uniform(100, 200),
            anim_offset=uniform(0, 1),
            anim_seed=int(np.random.randint(1e5)),
            z_scale=z_scale,
        )

    def apply_parameters(
        self, params: KelpMonocotParameters, *, spawn_scope: bool = True
    ) -> None:
        self._sample_material(params.seed)
        with FixedSeed(params.seed):
            self.align_angle = uniform(np.pi / 30, np.pi / 15)
        self.stem_offset = 10.0
        self.angle = params.angle
        self.z_drag = params.z_drag
        self.min_y_angle = params.min_y_angle
        self.max_y_angle = params.min_y_angle
        self.bend_angle = params.bend_angle
        self.twist_angle = params.twist_angle
        self.count = 512
        self.leaf_prob = params.leaf_prob
        self.radius = 0.02
        self.leaf_range = (0, 1)
        self.scale_curve = [(0, 1), (1, 1)]
        self.perturb = 0.05
        self.z_scale = params.z_scale
        self.align_direction = (
            np.cos(params.flow_angle),
            np.sin(params.flow_angle),
            params.align_direction_z,
        )
        self.anim_freq = 1 / params.anim_period
        self.anim_offset = params.anim_offset
        self.anim_seed = params.anim_seed
        self.align_factor = self.make_align_factor()
        self._use_fixed_spawn_draws = spawn_scope

    def make_align_factor(self):
        def align_factor(nw: NodeWrangler):
            rand = nw.uniform(0.7, 0.95)
            driver = rand.inputs[2].driver_add("default_value").driver
            driver.expression = repeated_driver(
                0.7, 0.85, self.anim_freq, self.anim_offset, self.anim_seed
            )
            return nw.scalar_multiply(nw.bernoulli(0.9), rand)

        return align_factor

    def make_align_direction(self):
        def align_direction(nw: NodeWrangler):
            direction = nw.combine(1, 0, 0)
            driver = direction.inputs[2].driver_add("default_value").driver
            driver.expression = repeated_driver(
                -0.5, -0.1, self.anim_freq, self.anim_offset, self.anim_seed
            )
            return direction

        return align_direction

    @staticmethod
    def build_base_hue():
        return uniform(0.05, 0.25)

    def build_instance(self, i, face_size):
        x_anchors = np.array([0, -0.02, -0.04])
        y_anchors = np.array([0, uniform(0.01, 0.02), 0])
        curves = []
        for angle in np.linspace(0, np.pi * 2, 6):
            anchors = [x_anchors, np.cos(angle) * y_anchors, np.sin(angle) * y_anchors]
            curves.append(bezier_curve(anchors))
        bud = butil.join_objects(curves)
        bud.location[0] += 0.02
        with butil.ViewportMode(bud, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.convex_hull()
        remesh_with_attrs(bud, face_size)

        x_anchors = 0, uniform(0.35, 0.65), uniform(0.8, 1.2)
        y_anchors = 0, uniform(0.06, 0.08), 0
        obj = leaf(x_anchors, y_anchors, face_size=face_size)
        obj = join_objects([obj, bud])
        self.decorate_leaf(
            obj,
            uniform(-2, 2),
            uniform(-np.pi / 4, np.pi / 4),
            uniform(-np.pi / 4, np.pi / 4),
        )
        origin2leftmost(obj)
        return obj

    def create_asset(self, **params):
        obj = self.create_raw(**params)
        self.decorate_monocot(obj)
        assign_material(obj, self.material)
        return obj
