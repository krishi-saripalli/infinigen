# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei

from __future__ import annotations

from typing import Annotated, ClassVar

import bmesh
import numpy as np
from numpy.random import uniform
from pydantic import Field

from infinigen.assets.objects.monocot.growth import MonocotGrowthFactory
from infinigen.assets.utils.decorate import displace_vertices, read_co
from infinigen.assets.utils.draw import bezier_curve, leaf
from infinigen.assets.utils.nodegroup import geo_radius
from infinigen.assets.utils.object import join_objects, origin2lowest
from infinigen.assets.utils.shapes import point_normal_up
from infinigen.core import surface
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


class BananaMonocotParameters(AssetParameters):
    stem_offset: Annotated[float, Field(ge=0.6, le=1.0, json_schema_extra={"editable": True})]
    angle: Annotated[float, Field(ge=0.785398, le=1.047198, json_schema_extra={"editable": True})]
    z_scale: Annotated[float, Field(ge=1.0, le=1.5, json_schema_extra={"editable": True})]
    z_drag: Annotated[float, Field(ge=0.1, le=0.2, json_schema_extra={"editable": True})]
    min_y_angle: Annotated[
        float, Field(ge=0.15708, le=0.314159, json_schema_extra={"editable": True})
    ]
    max_y_angle: Annotated[
        float, Field(ge=0.785398, le=1.413717, json_schema_extra={"editable": True})
    ]
    leaf_range_low: Annotated[
        float, Field(ge=0.5, le=0.7, json_schema_extra={"editable": True})
    ]
    count: Annotated[float, Field(ge=16.0, le=24.0, json_schema_extra={"editable": True})]
    scale_curve_low: Annotated[
        float, Field(ge=0.4, le=1.0, json_schema_extra={"editable": True})
    ]
    scale_curve_high: Annotated[
        float, Field(ge=0.6, le=1.0, json_schema_extra={"editable": True})
    ]
    radius: Annotated[float, Field(ge=0.04, le=0.06, json_schema_extra={"editable": True})]
    bud_angle: Annotated[
        float, Field(ge=0.392699, le=0.523599, json_schema_extra={"editable": True})
    ]
    cut_angle_offset: Annotated[
        float, Field(ge=0.15708, le=0.261799, json_schema_extra={"editable": True})
    ]
    freq: Annotated[float, Field(ge=100.0, le=300.0, json_schema_extra={"editable": True})]
    n_cuts_draw: Annotated[
        float,
        Field(ge=0.0, le=1.0, json_schema_extra={"editable": True, "kind": "draw_bool"}),
    ] = 0.0
    base_hue: Annotated[float, Field(ge=0.15, le=0.35, json_schema_extra={"editable": True})]
    leaf_prob: Annotated[float, Field(ge=0.8, le=0.9, json_schema_extra={"editable": True})]

class BananaMonocotFactory(ParameterizedAssetFactory, MonocotGrowthFactory):
    parameters_model: ClassVar[type[AssetParameters]] = BananaMonocotParameters

    def __init__(self, factory_seed, coarse=False):
        AssetFactory.__init__(self, factory_seed, coarse)
        self.init_legacy_parameters()
    def _sample_init_parameters(self, seed: int) -> BananaMonocotParameters:
        base_hue = uniform(0.15, 0.35)
        leaf_prob = uniform(0.8, 0.9)
        bright_color = hsv2rgba(base_hue, uniform(0.6, 0.8), log_uniform(0.05, 0.1))
        dark_color = hsv2rgba(
            (base_hue + uniform(-0.03, 0.03)) % 1,
            uniform(0.8, 1.0),
            log_uniform(0.05, 0.2),
        )
        self.material = shaderfunc_to_material(
            self.shader_monocot, dark_color, bright_color, self.use_distance
        )
        bud_angle = uniform(np.pi / 8, np.pi / 6)
        return BananaMonocotParameters(
            seed=seed,
            stem_offset=uniform(0.6, 1.0),
            angle=uniform(np.pi / 4, np.pi / 3),
            z_scale=uniform(1, 1.5),
            z_drag=uniform(0.1, 0.2),
            min_y_angle=uniform(np.pi * 0.05, np.pi * 0.1),
            max_y_angle=uniform(np.pi * 0.25, np.pi * 0.45),
            leaf_range_low=uniform(0.5, 0.7),
            count=log_uniform(16, 24),
            scale_curve_low=uniform(0.4, 1.0),
            scale_curve_high=uniform(0.6, 1.0),
            radius=uniform(0.04, 0.06),
            bud_angle=bud_angle,
            cut_angle_offset=uniform(np.pi / 20, np.pi / 12),
            freq=log_uniform(100, 300),
            n_cuts_draw=0.0,
            base_hue=base_hue,
            leaf_prob=leaf_prob,
        )

    def apply_parameters(
        self, params: BananaMonocotParameters, *, spawn_scope: bool = True
    ) -> None:
        self.stem_offset = params.stem_offset
        self.angle = params.angle
        self.z_scale = params.z_scale
        self.z_drag = params.z_drag
        self.min_y_angle = params.min_y_angle
        self.max_y_angle = params.max_y_angle
        self.leaf_range = (params.leaf_range_low, 1)
        self.count = int(params.count)
        self.scale_curve = [(0, params.scale_curve_low), (1, params.scale_curve_high)]
        self.radius = params.radius
        self.bud_angle = params.bud_angle
        self.cut_angle = params.bud_angle + params.cut_angle_offset
        self.freq = params.freq
        self.n_cuts = (
            0
            if params.n_cuts_draw >= 0.8
            else int(np.random.randint(6, 10))
        )
        self.leaf_prob = params.leaf_prob
        self.base_hue = params.base_hue
        self.bend_angle = np.pi / 4
        self.twist_angle = np.pi / 6
        self.perturb = 0.05
        self.align_factor = 0
        self.align_direction = (1, 0, 0)
        self._use_fixed_spawn_draws = spawn_scope
        self._cache_decor_state(params.seed)

    @staticmethod
    def build_base_hue():
        return uniform(0.15, 0.35)

    def cut_leaf(self, obj):
        coords = read_co(obj)
        x, y, z = coords.T
        coords = coords[(np.abs(y) < 0.08) & (np.abs(y) > 0.01)]
        positive_coords = coords[coords.T[1] > 0]
        positive_coords = positive_coords[np.argsort(positive_coords[:, 0])]
        negative_coords = coords[coords.T[1] < 0]
        negative_coords = negative_coords[np.argsort(negative_coords[:, 0])]
        positive_coords = positive_coords[
            np.random.choice(len(positive_coords), self.n_cuts, replace=False)
        ]
        negative_coords = negative_coords[
            np.random.choice(len(negative_coords), self.n_cuts, replace=False)
        ]

        for (x1, y1, _), (x2, y2, _) in zip(
            np.concatenate([positive_coords[:-1], negative_coords[:-1]], 0),
            np.concatenate([positive_coords[1:], negative_coords[1:]], 0),
        ):
            coeff = 1 if y1 > 0 else -1
            ratio = uniform(-2.0, 0.4)
            exponent = uniform(1.2, 1.6)

            def cut(x, y, z):
                m1 = x1 * np.sin(self.cut_angle) - y1 * np.cos(self.cut_angle) * coeff
                m2 = x2 * np.sin(self.cut_angle) - y2 * np.cos(self.cut_angle) * coeff
                m = x * np.sin(self.cut_angle) - y * np.cos(self.cut_angle) * coeff
                dist = ((x - x1) * (y1 - y2) + (y - y1) * (x1 - x2)) / np.sqrt(
                    (x1 - x2) ** 2 + (y1 - y2) ** 2 + 0.1
                )
                return (
                    0,
                    0,
                    np.where(
                        (m1 < m) & (m < m2) & (dist * coeff < 0),
                        ratio * np.abs(dist) ** exponent,
                        0,
                    ),
                )

            displace_vertices(obj, cut)
        with butil.ViewportMode(obj, "EDIT"):
            bm = bmesh.from_edit_mesh(obj.data)
            geom = [e for e in bm.edges if e.calc_length() > 0.02]
            bmesh.ops.delete(bm, geom=geom, context="EDGES")
            bmesh.update_edit_mesh(obj.data)

    def build_leaf(self, face_size):
        x_anchors = 0, 0.2 * np.cos(self.bud_angle), uniform(0.8, 1.2), 2.0
        y_anchors = 0, 0.2 * np.sin(self.bud_angle), uniform(0.2, 0.25), 0
        obj = leaf(x_anchors, y_anchors, face_size=face_size)
        self.cut_leaf(obj)
        self.displace_veins(obj)
        self.decorate_leaf(obj)
        tag_object(obj, "banana")
        return obj

    def displace_veins(self, obj):
        vg = obj.vertex_groups.new(name="distance")
        x, y, z = read_co(obj).T
        branch = np.cos(
            (np.abs(y) * np.cos(self.cut_angle) - x * np.sin(self.cut_angle))
            * self.freq
        ) > uniform(0.85, 0.9, len(x))
        leaf = np.abs(y) < uniform(0.002, 0.008, len(x))
        weights = branch | leaf
        for i, l in enumerate(weights):
            vg.add([i], l, "REPLACE")
        butil.modify_mesh(
            obj,
            "DISPLACE",
            strength=-uniform(5e-3, 8e-3),
            mid_level=0,
            vertex_group="distance",
        )


class TaroMonocotParameters(AssetParameters):
    stem_offset: Annotated[float, Field(ge=0.05, le=0.1, json_schema_extra={"editable": False})]
    bud_angle: Annotated[
        float, Field(ge=1.884956, le=2.199115, json_schema_extra={"editable": True})
    ]
    count: Annotated[float, Field(ge=12.0, le=16.0, json_schema_extra={"editable": True})]
    min_y_angle: Annotated[
        float, Field(ge=-0.785398, le=-0.15708, json_schema_extra={"editable": True})
    ]
    max_y_angle: Annotated[
        float, Field(ge=-0.15708, le=0.0, json_schema_extra={"editable": True})
    ]
    angle: Annotated[float, Field(ge=0.785398, le=1.047198, json_schema_extra={"editable": True})]
    z_scale: Annotated[float, Field(ge=1.0, le=1.5, json_schema_extra={"editable": True})]
    leaf_range_low: Annotated[
        float, Field(ge=0.5, le=0.7, json_schema_extra={"editable": True})
    ]
    scale_curve_low: Annotated[
        float, Field(ge=0.4, le=1.0, json_schema_extra={"editable": True})
    ]
    scale_curve_high: Annotated[
        float, Field(ge=0.6, le=1.0, json_schema_extra={"editable": True})
    ]
    leaf_prob: Annotated[float, Field(ge=0.8, le=0.9, json_schema_extra={"editable": False})]


class TaroMonocotFactory(BananaMonocotFactory):
    parameters_model: ClassVar[type[AssetParameters]] = TaroMonocotParameters

    def __init__(self, factory_seed, coarse=False):
        AssetFactory.__init__(self, factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_material(self, seed: int) -> None:
        # NOTE: base_hue is sampled on self in apply_parameters; excluded from quartet sampling (material-only, not exported geometry).
        with FixedSeed(seed):
            base_hue = uniform(0.15, 0.35)
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

    def _sample_init_parameters(self, seed: int) -> TaroMonocotParameters:
        leaf_prob = uniform(0.8, 0.9)
        self._sample_material(seed)
        bud_angle = uniform(np.pi * 0.6, np.pi * 0.7)
        return TaroMonocotParameters(
            seed=seed,
            stem_offset=uniform(0.05, 0.1),
            bud_angle=bud_angle,
            count=log_uniform(12, 16),
            min_y_angle=uniform(-np.pi * 0.25, -np.pi * 0.05),
            max_y_angle=uniform(-np.pi * 0.05, 0),
            angle=uniform(np.pi / 4, np.pi / 3),
            z_scale=uniform(1, 1.5),
            leaf_range_low=uniform(0.5, 0.7),
            scale_curve_low=uniform(0.4, 1.0),
            scale_curve_high=uniform(0.6, 1.0),
            leaf_prob=leaf_prob,
        )

    def apply_parameters(
        self, params: TaroMonocotParameters, *, spawn_scope: bool = True
    ) -> None:
        self._sample_material(params.seed)
        # NOTE: leaf_prob gates probabilistic leaf spawning per instance; intermittent across seeds, excluded from quartet sampling.
        self.stem_offset = params.stem_offset
        # NOTE: radius, freq, cut_angle_offset, n_cuts, and z_drag do not elicit a reliable visual change in exported geometry; sampled on self from seed, excluded from quartet sampling.
        with FixedSeed(params.seed):
            self.radius = uniform(0.02, 0.04)
            self.freq = log_uniform(10, 20)
            self.cut_angle_offset = uniform(np.pi / 20, np.pi / 12)
            self.n_cuts = int(np.random.randint(1, 2))
            self.z_drag = uniform(0.2, 0.3)
        self.bud_angle = params.bud_angle
        self.count = int(params.count)
        self.min_y_angle = params.min_y_angle
        self.max_y_angle = params.max_y_angle
        self.angle = params.angle
        self.z_scale = params.z_scale
        self.leaf_range = (params.leaf_range_low, 1)
        self.scale_curve = [(0, params.scale_curve_low), (1, params.scale_curve_high)]
        self.cut_angle = params.bud_angle + self.cut_angle_offset
        self.leaf_prob = params.leaf_prob
        self.bend_angle = np.pi / 4
        self.twist_angle = np.pi / 6
        self.perturb = 0.05
        self.align_factor = 0
        self.align_direction = (1, 0, 0)
        self._use_fixed_spawn_draws = spawn_scope
        self._cache_decor_state(params.seed)

    def displace_veins(self, obj):
        point_normal_up(obj)
        vg = obj.vertex_groups.new(name="distance")
        x, y, z = read_co(obj).T
        branch = np.cos(
            (np.abs(y) * np.cos(self.cut_angle) - x * np.sin(self.cut_angle))
            * self.freq
        ) > uniform(0.85, 0.9, len(x))
        leaf = np.abs(y) < uniform(0.002, 0.008, len(x))
        weights = branch | leaf
        for i, l in enumerate(weights):
            vg.add([i], l, "REPLACE")
        butil.modify_mesh(
            obj,
            "DISPLACE",
            strength=-uniform(5e-3, 8e-3),
            mid_level=0,
            vertex_group="distance",
        )

    def build_leaf(self, face_size):
        x_anchors = (
            0,
            0.2 * np.cos(self.bud_angle),
            max(self.radius * 12, 0.4),
            max(self.radius * 20, 0.8),
        )
        y_anchors = 0, 0.2 * np.sin(self.bud_angle), uniform(0.25, 0.3), 0
        obj = leaf(x_anchors, y_anchors, face_size=face_size)
        self.cut_leaf(obj)
        self.displace_veins(obj)
        self.decorate_leaf(obj, 2, leftmost=False)
        bezier = self.build_branch()
        obj = join_objects([obj, bezier])
        origin2lowest(obj)
        tag_object(obj, "taro")
        return obj

    def build_branch(self):
        offset = uniform(0.2, 0.3)
        length = uniform(1, 2)
        x_anchors = 0, -0.05, -offset - uniform(0.01, 0.02), -offset
        z_anchors = 0, 0, -length + 0.1, -length
        bezier = bezier_curve([x_anchors, 0, z_anchors])
        surface.add_geomod(
            bezier, geo_radius, apply=True, input_args=[uniform(0.02, 0.03), 32]
        )
        return bezier

    def build_instance(self, i, face_size):
        return self.build_leaf(face_size)
