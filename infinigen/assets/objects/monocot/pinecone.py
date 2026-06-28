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
from infinigen.assets.objects.monocot.growth import MonocotGrowthFactory
from infinigen.assets.utils.draw import shape_by_angles, shape_by_xs
from infinigen.assets.utils.object import new_circle
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_utils import build_color_ramp
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.placement.detail import remesh_with_attrs
from infinigen.core.placement.parameters import (
    AssetParameters,
    ParameterizedAssetFactory,
)
from infinigen.core.surface import shaderfunc_to_material
from infinigen.core.tagging import tag_object
from infinigen.core.util.color import hsv2rgba
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform


class PineconeParameters(AssetParameters):
    angle: Annotated[float, Field(ge=0.739008, le=1.396263, json_schema_extra={"editable": True})]
    count: Annotated[float, Field(ge=64.0, le=96.0, json_schema_extra={"editable": True})]
    stem_offset: Annotated[
        float, Field(ge=0.2, le=0.4, json_schema_extra={"editable": False})
    ]
    scale_curve_mid: Annotated[
        float, Field(ge=0.6, le=1.0, json_schema_extra={"editable": True})
    ]
    z_scale: Annotated[float, Field(ge=1.0, le=1.2, json_schema_extra={"editable": True})]


class PineconeFactory(ParameterizedAssetFactory, MonocotGrowthFactory):
    parameters_model: ClassVar[type[AssetParameters]] = PineconeParameters

    def __init__(self, factory_seed, coarse=False):
        super(PineconeFactory, self).__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_material(self, seed: int) -> None:
        # NOTE: color_hue is sampled on self in apply_parameters; excluded from quartet sampling (material-only, not exported geometry).
        with FixedSeed(seed):
            color_hue = uniform(0.02, 0.06)
            bright_color = hsv2rgba(color_hue, uniform(0.8, 1.0), 0.01)
            dark_color = hsv2rgba(color_hue, uniform(0.8, 1.0), 0.005)
        self.material = shaderfunc_to_material(
            self.shader_monocot, dark_color, bright_color, self.use_distance
        )

    def _sample_init_parameters(self, seed: int) -> PineconeParameters:
        z_scale = uniform(1.0, 1.2)
        self._sample_material(seed)
        scale_curve_mid = uniform(0.6, 1.0)
        return PineconeParameters(
            seed=seed,
            angle=2 * np.pi / (np.random.randint(4, 8) + 0.5),
            count=log_uniform(64, 96),
            stem_offset=uniform(0.2, 0.4),
            scale_curve_mid=scale_curve_mid,
            z_scale=z_scale,
        )

    def apply_parameters(
        self, params: PineconeParameters, *, spawn_scope: bool = True
    ) -> None:
        self._sample_material(params.seed)
        self.angle = params.angle
        # NOTE: max_y_angle, leaf_prob, scale_curve_high do not elicit a clear visual change in exported geometry; sampled on self, excluded from quartet sampling.
        with FixedSeed(params.seed):
            self.max_y_angle = uniform(0.7, 0.8) * np.pi / 2
            self.leaf_prob = uniform(0.9, 0.95)
            self.scale_curve_high = uniform(0.1, 0.2)
        self.min_y_angle = 0.0
        self.count = int(params.count)
        # NOTE: stem_offset sets stem length; scale on curve is gated by leaf_prob when instancing scales.
        self.stem_offset = params.stem_offset
        self.perturb = 0
        self.scale_curve = [
            (0, 0.5),
            (0.5, params.scale_curve_mid),
            (1, self.scale_curve_high),
        ]
        self.leaf_range = (0, 1)
        self.radius = 0.01
        self.bend_angle = np.pi / 4
        self.twist_angle = np.pi / 6
        self.z_drag = 0.0
        self.z_scale = params.z_scale
        self.align_factor = 0
        self.align_direction = (1, 0, 0)
        self._use_fixed_spawn_draws = spawn_scope
        self._cache_decor_state(params.seed)

    def build_leaf(self, face_size):
        obj = new_circle(vertices=128)
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.fill_grid()
        angles = np.array([-1, -0.8, -0.5, 0, 0.5, 0.8, 1]) * self.angle / 2
        scale = uniform(0.9, 0.95)
        scales = [0, 0.7, scale, 1, scale, 0.7, 0]
        displacement = [0, 0, 0, -uniform(0.2, 0.3), 0, 0, 0]
        shape_by_angles(obj, angles, scales, displacement)

        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.convex_hull()

        xs = [0, 1, 2]
        displacement = [0, 0, 0.5]
        shape_by_xs(obj, xs, displacement)

        obj.scale = [0.1] * 3
        obj.rotation_euler[1] -= uniform(np.pi / 18, np.pi / 12)
        butil.apply_transform(obj)
        remesh_with_attrs(obj, face_size)

        texture = bpy.data.textures.new(name="pinecone", type="STUCCI")
        texture.noise_scale = log_uniform(0.002, 0.005)
        butil.modify_mesh(
            obj, "DISPLACE", True, strength=0.001, mid_level=0, texture=texture
        )

        tag_object(obj, "pinecone")
        return obj

    @staticmethod
    def shader_monocot(nw: NodeWrangler, dark_color, bright_color, use_distance):
        specular = uniform(0.2, 0.4)
        color = build_color_ramp(
            nw,
            nw.musgrave(10),
            [0.0, 0.3, 0.7, 1.0],
            [bright_color, bright_color, dark_color, dark_color],
        )
        noise_texture = nw.new_node(Nodes.NoiseTexture, input_kwargs={"Scale": 50})
        roughness = nw.build_float_curve(noise_texture, [(0, 0.5), (1, 0.8)])
        bsdf = nw.new_node(
            Nodes.PrincipledBSDF,
            input_kwargs={
                "Base Color": color,
                "Roughness": roughness,
                "Specular IOR Level": specular,
            },
        )
        return bsdf
