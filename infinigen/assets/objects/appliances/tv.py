# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Lingjie Mei: primary author
# - Karhan Kayan: fix rotation

from __future__ import annotations

from typing import Annotated, ClassVar

import bmesh
import bpy
import numpy as np
from numpy.random import uniform
from pydantic import Field

from infinigen.assets.composition import material_assignments
from infinigen.assets.materials.text import Text
from infinigen.assets.utils.decorate import (
    mirror,
    read_area,
    read_co,
    read_normal,
    write_attribute,
    write_co,
)
from infinigen.assets.utils.nodegroup import geo_radius
from infinigen.assets.utils.object import (
    data2mesh,
    join_objects,
    mesh2obj,
    new_bbox,
    new_cube,
    new_plane,
)
from infinigen.assets.utils.uv import (
    compute_uv_direction,
    face_corner2faces,
    unwrap_faces,
)
from infinigen.core import surface
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import (
    AssetParameters,
    ParameterizedAssetFactory,
)
from infinigen.core.surface import write_attr_data
from infinigen.core.util import blender as butil
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform, weighted_sample


class TVParameters(AssetParameters):
    width: Annotated[float, Field(ge=0.6, le=2.1, json_schema_extra={"editable": True})]
    screen_bevel_width: Annotated[
        float, Field(ge=0.0, le=0.01, json_schema_extra={"editable": True})
    ]
    side_margin: Annotated[float, Field(ge=0.005, le=0.01, json_schema_extra={"editable": True})]
    depth: Annotated[float, Field(ge=0.02, le=0.04, json_schema_extra={"editable": True})]
    depth_extrude_multiplier: Annotated[
        float, Field(ge=2.0, le=5.0, json_schema_extra={"editable": True})
    ]
    leg_length: Annotated[float, Field(ge=0.1, le=0.2, json_schema_extra={"editable": True})]
    leg_width: Annotated[float, Field(ge=0.5, le=0.8, json_schema_extra={"editable": False})]
    tv_164: Annotated[float, Field(ge=0.1, le=0.3, json_schema_extra={"editable": False})] = (
        0.2
    )
    tv_165: Annotated[float, Field(ge=0.5, le=0.7, json_schema_extra={"editable": False})] = (
        0.6
    )
    tv_176: Annotated[float, Field(ge=0.0, le=0.4, json_schema_extra={"editable": False})] = (
        0.2
    )
    tv_241: Annotated[float, Field(ge=0.0, le=0.6, json_schema_extra={"editable": False})] = (
        0.3
    )
    tv_243: Annotated[float, Field(ge=0.3, le=0.5, json_schema_extra={"editable": False})] = (
        0.4
    )
    tv_250: Annotated[float, Field(ge=0.0, le=0.6, json_schema_extra={"editable": False})] = (
        0.3
    )
    width_1: Annotated[float, Field(ge=0.3, le=0.6, json_schema_extra={"editable": False})] = (
        0.45
    )
    leg_style: Annotated[
        str,
        Field(
            json_schema_extra={
                "editable": True,
                "kind": "enum",
                "choices": ["two-legged", "single-legged"],
            }
        ),
    ] = "two-legged"
    aspect_ratio_preset: Annotated[
        str,
        Field(
            json_schema_extra={
                "editable": True,
                "kind": "enum",
                "choices": ["16:9", "4:3"],
            }
        ),
    ] = "16:9"


class TVFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = TVParameters
    _aspect_ratio_by_preset = {"16:9": 9 / 16, "4:3": 3 / 4}

    def __init__(self, factory_seed, coarse=False):
        super(TVFactory, self).__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> TVParameters:
        depth = uniform(0.02, 0.04)
        screen_surface = weighted_sample(material_assignments.graphicdesign)()
        screen_emission = 0.01
        if isinstance(screen_surface, Text):
            screen_emission = 0.01 if uniform() < 0.1 else uniform(2, 3)
        aspect_ratio_preset = str(np.random.choice(["16:9", "4:3"]))
        self._aspect_ratio = self._aspect_ratio_by_preset[aspect_ratio_preset]
        self._leg_type = np.random.choice(["two-legged", "single-legged"])
        self._surface = weighted_sample(material_assignments.metal_neutral)()
        self._support_surface = weighted_sample(material_assignments.metal_neutral)()
        self._screen_surface = screen_surface
        self._screen_emission = screen_emission
        return TVParameters(
            seed=seed,
            width=uniform(0.6, 2.1),
            screen_bevel_width=uniform(0, 0.01),
            side_margin=log_uniform(0.005, 0.01),
            depth=depth,
            depth_extrude_multiplier=uniform(2, 5),
            leg_length=uniform(0.1, 0.2),
            leg_width=uniform(0.5, 0.8),
            tv_164=uniform(0.1, 0.3),
            tv_165=uniform(0.5, 0.7),
            tv_176=uniform(0.0, 0.4),
            tv_241=uniform(0.0, 0.6),
            tv_243=uniform(0.3, 0.5),
            tv_250=uniform(0.0, 0.6),
            width_1=uniform(0.3, 0.6),
            leg_style=self._leg_type,
            aspect_ratio_preset=aspect_ratio_preset,
        )

    def _sample_spawn_parameters(
        self, params: TVParameters, seed: int, i: int
    ) -> TVParameters:
        return params

    def apply_parameters(
        self, params: TVParameters, *, spawn_scope: bool = True
    ) -> None:
        if not hasattr(self, "_aspect_ratio"):
            aspect_ratio_preset = str(np.random.choice(["16:9", "4:3"]))
            self._aspect_ratio = self._aspect_ratio_by_preset[aspect_ratio_preset]
            self._leg_type = np.random.choice(["two-legged", "single-legged"])
            self._surface = weighted_sample(material_assignments.metal_neutral)()
            self._support_surface = weighted_sample(material_assignments.metal_neutral)()
            screen_surface = weighted_sample(material_assignments.graphicdesign)()
            self._screen_emission = 0.01
            if isinstance(screen_surface, Text):
                self._screen_emission = 0.01 if uniform() < 0.1 else uniform(2, 3)
            self._screen_surface = screen_surface
        # NOTE: bottom_margin, leg_radius, leg_bevel_width sampled on self from seed; excluded from quartet sampling.
        with FixedSeed(params.seed):
            self.bottom_margin = uniform(0.005, 0.03)
            self.leg_radius = uniform(0.008, 0.015)
            self.leg_bevel_width = uniform(0.01, 0.02)
            self.leg_length_y = uniform(0.1, 0.15)
        self.leg_width = params.leg_width
        self.aspect_ratio = self._aspect_ratio_by_preset[params.aspect_ratio_preset]
        self.width = params.width
        self.screen_bevel_width = params.screen_bevel_width
        self.side_margin = params.side_margin
        self.depth = params.depth
        self.has_depth_extrude = False
        if self.has_depth_extrude:
            self.depth_extrude = self.depth * params.depth_extrude_multiplier
        else:
            self.depth_extrude = self.depth * 1.5
        self.leg_type = params.leg_style
        self.leg_length = params.leg_length
        self.surface = self._surface
        self.support_surface = self._support_surface
        self.screen_surface = self._screen_surface
        if isinstance(self.screen_surface, Text):
            self.screen_surface.emission = self._screen_emission
        self._use_fixed_spawn_draws = spawn_scope
        if spawn_scope:
            self._tv_164 = params.tv_164
            self._tv_165 = params.tv_165
            self._tv_176 = params.tv_176
            self._tv_241 = params.tv_241
            self._tv_243 = params.tv_243
            self._tv_250 = params.tv_250
            self._width_1 = params.width_1

    @property
    def height(self):
        return self.aspect_ratio * self.width

    @property
    def total_width(self):
        return self.width + 2 * self.side_margin

    @property
    def total_height(self):
        return self.height + self.side_margin + self.bottom_margin

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        match self.leg_type:
            case "two-legged":
                max_x = (
                    self.leg_length_y / 2 - (1 - self.leg_width) * self.depth_extrude
                )
            case _:
                max_x = self.leg_length_y / 2 - self.depth_extrude / 2
        return new_bbox(
            -self.depth_extrude - self.depth,
            max_x,
            -self.total_width / 2,
            self.total_width / 2,
            -self.leg_length - self.leg_radius / 2,
            self.total_height,
        )

    def create_asset(self, **params) -> bpy.types.Object:
        obj = self.make_base()
        self.make_screen(obj)
        parts = [obj]
        match self.leg_type:
            case "two-legged":
                legs = self.add_two_legs()
            case _:
                legs = self.add_single_leg()
        for leg_obj in legs:
            write_attribute(leg_obj, 1, "leg", "FACE", "INT")
        parts.extend(legs)
        obj = join_objects(parts)

        surface.assign_material(obj, self.surface())
        surface.assign_material(obj, self.support_surface(), selection="leg")
        surface.assign_material(obj, self.screen_surface(), selection="screen")

        obj.rotation_euler[2] = np.pi / 2
        butil.apply_transform(obj)
        return obj

    def make_screen(self, obj):
        cutter = new_cube()
        cutter.location = 0, -1, 1
        butil.apply_transform(cutter, True)
        cutter.scale = self.width / 2, 1, self.height / 2
        cutter.location = 0, 1e-3, self.bottom_margin
        butil.apply_transform(cutter, True)
        butil.modify_mesh(obj, "BOOLEAN", object=cutter, operation="DIFFERENCE")
        butil.delete(cutter)
        areas = read_area(obj)
        screen = np.zeros(len(areas), int)
        y = read_normal(obj)[:, 1] < 0
        screen[np.argmax(areas + 1e5 * y)] = 1
        fc2f = face_corner2faces(obj)
        unwrap_faces(obj, screen)
        bbox = compute_uv_direction(obj, "x", "z", screen[fc2f])
        write_attr_data(obj, "screen", screen, domain="FACE", type="INT")

    def make_base(self):
        obj = new_cube()
        obj.location = 0, 1, 1
        butil.apply_transform(obj, True)
        obj.scale = self.total_width / 2, self.depth / 2, self.total_height / 2
        butil.apply_transform(obj)
        butil.modify_mesh(obj, "BEVEL", width=self.screen_bevel_width, segments=8)
        if not self.has_depth_extrude:
            return obj
        with butil.ViewportMode(obj, "EDIT"):
            bm = bmesh.from_edit_mesh(obj.data)
            geom = [f for f in bm.faces if f.normal[1] > 0.5]
            bmesh.ops.delete(bm, geom=geom, context="FACES_KEEP_BOUNDARY")
            bmesh.update_edit_mesh(obj.data)
            bpy.ops.mesh.select_mode(type="EDGE")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.region_to_loop()
        height_min, height_max = (
            self.total_height * (
                self._tv_164 if self._use_fixed_spawn_draws else uniform(0.1, 0.3)
            ),
            self.total_height * (
                self._tv_165 if self._use_fixed_spawn_draws else uniform(0.5, 0.7)
            ),
        )
        width = self.total_width * (
            self._width_1 if self._use_fixed_spawn_draws else uniform(0.3, 0.6)
        )
        extra = new_plane()
        extra.scale = width / 2, (height_max - height_min) / 2, 1
        extra.rotation_euler[0] = -np.pi / 2
        extra.location = 0, self.depth_extrude + self.depth, self.total_height / 2
        obj = join_objects([obj, extra])
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_mode(type="EDGE")
            bpy.ops.mesh.bridge_edge_loops(
                number_cuts=32,
                profile_shape_factor=-(
                    self._tv_176 if self._use_fixed_spawn_draws else uniform(0.0, 0.4)
                ),
            )
        x, y, z = read_co(obj).T
        z += (
            (height_max + height_min - self.total_height)
            / 2
            * np.clip(y - self.depth, 0, None)
            / self.depth_extrude
        )
        write_co(obj, np.stack([x, y, z], -1))
        return obj

    def add_two_legs(self):
        leg_x_frac = self._tv_241 if self._use_fixed_spawn_draws else uniform(0, 0.6)
        leg_attach_z_frac = (
            self._tv_243 if self._use_fixed_spawn_draws else uniform(0.3, 0.5)
        )
        leg_z_clip_frac = (
            self._tv_250 if self._use_fixed_spawn_draws else uniform(0.0, 0.6)
        )
        vertices = (
            (
                -self.total_width / 2 * self.leg_width * leg_x_frac,
                0,
                self.total_height * leg_attach_z_frac,
            ),
            (0, 0, -self.leg_length),
            (0, self.leg_length_y / 2, -self.leg_length),
            (0, -self.leg_length_y / 2, -self.leg_length),
        )
        edges = (0, 1), (1, 2), (1, 3)
        leg = mesh2obj(data2mesh(vertices, edges))
        surface.add_geomod(
            leg, geo_radius, apply=True, input_args=[self.leg_radius, 16]
        )
        x, y, z = read_co(leg).T
        write_co(
            leg,
            np.stack(
                [
                    x,
                    y,
                    np.maximum(
                        z, -self.leg_length - self.leg_radius * leg_z_clip_frac
                    ),
                ],
                -1,
            ),
        )
        leg_ = deep_clone_obj(leg)
        butil.select_none()
        leg.location = (
            self.total_width / 2 * self.leg_width,
            (1 - self.leg_width) * self.depth_extrude,
            0,
        )
        butil.apply_transform(leg, True)
        mirror(leg_)
        leg_.location = (
            -self.total_width / 2 * self.leg_width,
            (1 - self.leg_width) * self.depth_extrude,
            0,
        )
        butil.apply_transform(leg_, True)
        return [leg, leg_]

    def add_single_leg(self):
        leg_attach_z_frac = (
            self._tv_243 if self._use_fixed_spawn_draws else uniform(0.3, 0.5)
        )
        leg = new_cube()
        leg.location = 0, 1, 1
        butil.apply_transform(leg, True)
        leg.location = 0, self.depth_extrude / 2, -self.leg_length
        leg.scale = [
            self.total_width * uniform(0.05, 0.1),
            self.leg_radius,
            (self.leg_length + self.total_height * leg_attach_z_frac) / 2,
        ]
        butil.apply_transform(leg, True)
        butil.modify_mesh(leg, "BEVEL", width=self.leg_bevel_width, segments=8)
        base = new_cube()
        base.location = 0, self.depth_extrude / 2, -self.leg_length
        base.scale = [
            self.total_width * uniform(0.15, 0.3),
            self.leg_length_y / 2,
            self.leg_radius,
        ]
        butil.apply_transform(base, True)
        butil.modify_mesh(base, "BEVEL", width=self.leg_bevel_width, segments=8)
        return [leg, base]


class MonitorParameters(TVParameters):
    pass


class MonitorFactory(TVFactory):
    parameters_model: ClassVar[type[AssetParameters]] = MonitorParameters

    def _sample_init_parameters(self, seed: int) -> MonitorParameters:
        params = super()._sample_init_parameters(seed)
        return MonitorParameters.model_validate(
            params.model_copy(
                update={
                    "width": log_uniform(0.4, 0.8),
                    "leg_style": "single-legged",
                }
            ).model_dump()
        )
