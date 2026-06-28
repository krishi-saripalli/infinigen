# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Lingjie Mei

from __future__ import annotations

from typing import Annotated, ClassVar

import bpy
import numpy as np
from numpy.random import uniform
from pydantic import Field

from infinigen.assets.objects.monocot.growth import MonocotGrowthFactory
from infinigen.assets.utils.decorate import (
    distance2boundary,
    write_attribute,
    write_material_index,
)
from infinigen.assets.utils.draw import leaf, spin
from infinigen.assets.utils.misc import assign_material
from infinigen.assets.utils.object import join_objects
from infinigen.core import surface
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import (
    AssetParameters,
    ParameterizedAssetFactory,
)
from infinigen.core.surface import shaderfunc_to_material
from infinigen.core.tagging import tag_object
from infinigen.core.util import blender as butil
from infinigen.core.util.color import hsv2rgba
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform


class VeratrumMonocotParameters(AssetParameters):
    stem_offset: Annotated[float, Field(ge=1.0, le=1.5, json_schema_extra={"editable": True})]
    angle: Annotated[float, Field(ge=0.785398, le=1.047198, json_schema_extra={"editable": True})]
    z_drag: Annotated[float, Field(ge=0.4, le=0.5, json_schema_extra={"editable": False})]
    min_y_angle: Annotated[
        float, Field(ge=0.785398, le=1.099557, json_schema_extra={"editable": True})
    ]
    max_y_angle: Annotated[
        float, Field(ge=1.884956, le=2.199115, json_schema_extra={"editable": False})
    ]
    count: Annotated[float, Field(ge=32.0, le=64.0, json_schema_extra={"editable": True})]
    leaf_prob: Annotated[float, Field(ge=0.8, le=0.9, json_schema_extra={"editable": False})]


class VeratrumMonocotFactory(ParameterizedAssetFactory, MonocotGrowthFactory):
    parameters_model: ClassVar[type[AssetParameters]] = VeratrumMonocotParameters

    def __init__(self, factory_seed, coarse=False):
        AssetFactory.__init__(self, factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_material(self, seed: int) -> None:
        # NOTE: base_hue is sampled on self in apply_parameters; excluded from quartet sampling (material-only, not exported geometry).
        with FixedSeed(seed):
            base_hue = uniform(0.12, 0.32)
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

    def _sample_init_parameters(self, seed: int) -> VeratrumMonocotParameters:
        leaf_prob = uniform(0.8, 0.9)
        self._sample_material(seed)
        self.branch_material = shaderfunc_to_material(self.shader_ear)
        self.branches_factory = VeratrumBranchMonocotFactory(seed, self.coarse)
        return VeratrumMonocotParameters(
            seed=seed,
            stem_offset=uniform(1.0, 1.5),
            angle=uniform(np.pi / 4, np.pi / 3),
            z_drag=uniform(0.4, 0.5),
            min_y_angle=uniform(np.pi * 0.25, np.pi * 0.35),
            max_y_angle=uniform(np.pi * 0.6, np.pi * 0.7),
            count=log_uniform(32, 64),
            leaf_prob=leaf_prob,
        )

    def apply_parameters(
        self, params: VeratrumMonocotParameters, *, spawn_scope: bool = True
    ) -> None:
        self._sample_material(params.seed)
        # NOTE: scale_curve_0 and z_scale do not elicit a reliable visual change in exported geometry; sampled on self from seed, excluded from quartet sampling.
        with FixedSeed(params.seed):
            scale_curve_0 = uniform(0.8, 1.0)
            self.z_scale = uniform(1.0, 1.2)
            scale_curve_2 = uniform(0, 0.1)
            leaf_range_high = uniform(0.7, 0.8)
            self.bud_angle = uniform(np.pi / 15, np.pi / 12)
            self.freq = uniform(25, 50)
        self.stem_offset = params.stem_offset
        self.angle = params.angle
        self.z_drag = params.z_drag
        self.bend_angle = np.pi / 2
        self.min_y_angle = params.min_y_angle
        self.max_y_angle = params.max_y_angle
        self.count = int(params.count)
        self.scale_curve = (
            (0, scale_curve_0),
            (0.4, 0.6),
            (0.8, scale_curve_2),
            (1, 0),
        )
        self.leaf_range = (0, leaf_range_high)
        self.leaf_prob = params.leaf_prob
        self.radius = 0.01
        self.perturb = 0.05
        self.twist_angle = np.pi / 6
        self.align_factor = 0
        self.align_direction = (1, 0, 0)
        self._cache_decor_state(params.seed)
        self._use_fixed_spawn_draws = spawn_scope
        if hasattr(self, "branches_factory"):
            for ear_factory in self.branches_factory.branch_factories:
                ear_factory.scale_curve = self.scale_curve
                ear_factory.leaf_range = self.leaf_range
                ear_factory.z_scale = self.z_scale
                ear_factory._cache_decor_state(params.seed)

    @staticmethod
    def build_base_hue():
        return uniform(0.12, 0.32)

    @staticmethod
    def shader_ear(nw: NodeWrangler):
        color = hsv2rgba(uniform(0.1, 0.35), uniform(0.1, 0.5), log_uniform(0.2, 0.5))
        specular = uniform(0.0, 0.2)
        clearcoat = 0 if uniform(0, 1) < 0.8 else uniform(0.2, 0.5)
        noise_texture = nw.new_node(Nodes.NoiseTexture, input_kwargs={"Scale": 50})
        roughness = nw.build_float_curve(noise_texture, [(0, 0.5), (1, 0.8)])
        bsdf = nw.new_node(
            Nodes.PrincipledBSDF,
            input_kwargs={
                "Base Color": color,
                "Roughness": roughness,
                "Specular IOR Level": specular,
                "Coat Weight": clearcoat,
                "Subsurface Weight": 0.01,
                "Subsurface Radius": (0.01, 0.01, 0.01),
            },
        )
        return bsdf

    def build_leaf(self, face_size):
        x_anchors = 0, 0.2 * np.cos(self.bud_angle), uniform(0.6, 0.7), 0.8
        y_anchors = 0, 0.2 * np.sin(self.bud_angle), uniform(0.06, 0.1), 0
        obj = leaf(x_anchors, y_anchors, face_size=face_size)
        distance = distance2boundary(obj)

        vg = obj.vertex_groups.new(name="distance")
        weights = np.cos(self.freq * distance) ** 4
        for i, w in enumerate(weights):
            vg.add([i], w, "REPLACE")
        butil.modify_mesh(
            obj,
            "DISPLACE",
            strength=-uniform(5e-3, 8e-3),
            mid_level=0,
            vertex_group="distance",
        )
        self.decorate_leaf(obj, 8, np.pi / 2)
        return obj

    def create_asset(self, **params):
        obj = super().create_raw(**params)
        branches = self.branches_factory.create_asset(**params)
        branches.location[-1] = self.stem_offset - 0.02
        obj = join_objects([obj, branches])

        self.decorate_monocot(obj)
        assign_material(obj, [self.material, self.branch_material])
        write_material_index(
            obj, surface.read_attr_data(obj, "ear", "FACE").astype(int)
        )
        tag_object(obj, "veratrum")
        return obj


class VeratrumBranchMonocotFactory(AssetFactory):
    max_branches = 6

    def __init__(self, factory_seed, coarse=False):
        super(VeratrumBranchMonocotFactory, self).__init__(factory_seed, coarse)
        self.branch_factories = [
            VeratrumEarMonocotFactory(self.factory_seed * self.max_branches + i, coarse)
            for i in range(np.random.randint(3, self.max_branches) + 1)
        ]
        self.primary_stem_offset = uniform(0.4, 0.8)

        for i, f in enumerate(self.branch_factories):
            scale = log_uniform(0.3, 0.6) if i > 0 else 1
            f.stem_offset = scale * self.primary_stem_offset
            f.count = int(log_uniform(64, 238) * scale)

    def create_asset(self, **params) -> bpy.types.Object:
        branches = [f.create_asset(**params) for f in self.branch_factories]
        for i, branch in enumerate(branches):
            if i > 0:
                branch.location[-1] = self.primary_stem_offset * uniform(0, 0.6)
                branch.rotation_euler = (
                    uniform(np.pi * 0.25, np.pi * 0.4),
                    0,
                    uniform(0, np.pi * 2),
                )
        obj = join_objects(branches)
        tag_object(obj, "veratrum_branch")
        return obj


class VeratrumEarMonocotFactory(MonocotGrowthFactory):
    def __init__(self, factory_seed, coarse=False):
        super(VeratrumEarMonocotFactory, self).__init__(factory_seed, coarse)
        self.angle = uniform(np.pi / 4, np.pi / 3)
        self.min_y_angle = uniform(np.pi * 0.25, np.pi * 0.3)
        self.max_y_angle = uniform(np.pi * 0.3, np.pi * 0.35)
        self.count = np.random.randint(64, 128)
        self.leaf_prob = uniform(0.6, 0.8)
        self.leaf_range = 0, 0.98

    def build_leaf(self, face_size):
        x_anchors = 0, 0.04, 0.06, 0.04, 0
        y_anchors = 0, 0.01, 0, -0.01, 0
        z_anchors = 0, -0.01, -0.01, -0.006, 0
        anchors = [x_anchors, y_anchors, z_anchors]
        obj = spin(
            anchors,
            [0, 2, 4],
            dupli=True,
            loop=True,
            rotation_resolution=np.random.randint(3, 5),
            axis=(1, 0, 0),
        )
        butil.modify_mesh(obj, "WELD", merge_threshold=face_size / 2)
        write_attribute(obj, 1, "ear", "FACE")
        tag_object(obj, "veratrum_ear")
        return obj
