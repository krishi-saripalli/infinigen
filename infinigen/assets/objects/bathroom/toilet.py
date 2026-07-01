# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from __future__ import annotations

from typing import Annotated, Any, ClassVar

import bpy
import numpy as np
from numpy.random import uniform
from pydantic import Field

from infinigen.assets.composition import material_assignments
from infinigen.assets.utils.decorate import (
    read_center,
    read_co,
    read_edge_center,
    read_edges,
    read_normal,
    select_edges,
    select_faces,
    select_vertices,
    subsurf,
    write_attribute,
    write_co,
)
from infinigen.assets.utils.draw import align_bezier
from infinigen.assets.utils.object import join_objects, new_bbox, new_cube, new_cylinder
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import (
    AssetParameters,
    ParameterizedAssetFactory,
)
from infinigen.core.util import blender as butil
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.math import FixedSeed, normalize
from infinigen.core.util.random import log_uniform, weighted_sample


class ToiletParameters(AssetParameters):
    height_ratio: Annotated[float, Field(ge=0.8, le=0.9, json_schema_extra={"editable": False})]
    tank_height_ratio: Annotated[float, Field(ge=0.6, le=1.0, json_schema_extra={"editable": False})]
    tank_cap_extrude: Annotated[
        bool, Field(json_schema_extra={"editable": False, "kind": "bool"})
    ] = False
    cover_rotation: Annotated[
        float, Field(ge=0.0, le=1.570796, json_schema_extra={"editable": True})
    ]
    hardware_cap: Annotated[
        float,
        Field(
            ge=0.01,
            le=0.015,
            json_schema_extra={"editable": False},
        ),
    ]
    hardware_length: Annotated[
        float,
        Field(
            ge=0.04,
            le=0.05,
            json_schema_extra={"editable": False},
        ),
    ]
    hardware_on_side: Annotated[
        bool,
        Field(
            json_schema_extra={
                "editable": False,
                "kind": "bool",
            }
        ),
    ]
    flush_control: Annotated[
        str,
        Field(
            json_schema_extra={
                "editable": False,
                "kind": "enum",
                "choices": ["button", "handle"],
            }
        ),
    ] = "button"


class ToiletFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = ToiletParameters

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_runtime(self) -> tuple[Any, Any, Any, Any, tuple[float, float, float, float]]:
        scratch_prob, edge_wear_prob = material_assignments.wear_tear_prob
        scratch_fn, edge_wear_fn = material_assignments.wear_tear
        scratch_draw = uniform()
        edge_wear_draw = uniform()
        return (
            weighted_sample(material_assignments.ceramics)(),
            weighted_sample(material_assignments.metal_neutral)(),
            None if scratch_draw > scratch_prob else scratch_fn(),
            None if edge_wear_draw > edge_wear_prob else edge_wear_fn(),
            log_uniform(0.8, 1.2, 4),
        )

    def _sample_init_parameters(self, seed: int) -> ToiletParameters:
        (
            self._surface,
            self._hardware_surface,
            self._scratch,
            self._edge_wear,
            self._curve_scale,
        ) = self._sample_runtime()
        return ToiletParameters(
            seed=seed,
            height_ratio=uniform(0.8, 0.9),
            tank_height_ratio=uniform(0.6, 1.0),
            tank_cap_extrude=False,
            cover_rotation=uniform(0, np.pi / 2),
            hardware_cap=uniform(0.01, 0.015),
            hardware_length=uniform(0.04, 0.05),
            hardware_on_side=bool(uniform() < 0.5),
            flush_control=str(np.random.choice(["button", "handle"])),
        )

    def apply_parameters(
        self, params: ToiletParameters, *, spawn_scope: bool = True
    ) -> None:
        if not hasattr(self, "_surface"):
            (
                self._surface,
                self._hardware_surface,
                self._scratch,
                self._edge_wear,
                self._curve_scale,
            ) = self._sample_runtime()
        # NOTE: size sampled on self from seed; excluded from quartet sampling (uniform scale normalized away in point clouds).
        with FixedSeed(params.seed):
            self.size = uniform(0.4, 0.5)
        # NOTE: width_ratio, size_mid, depth_ratio, tube_scale, thickness, extrude_height, stand_depth_ratio, stand_scale, bottom_offset, back_thickness_ratio, back_size_ratio, back_scale, seat_thickness_ratio, seat_size_ratio, has_seat_cut, tank_width_ratio, tank_size_gap, tank_cap_height, tank_cap_extrude_amount, hardware_radius sampled on self from seed; excluded from quartet sampling.
        with FixedSeed(params.seed):
            self.width_ratio = uniform(0.7, 0.8)
            self.size_mid = uniform(0.6, 0.65)
            self.depth_ratio = uniform(0.5, 0.6)
            self.tube_scale = uniform(0.25, 0.3)
            self.thickness = uniform(0.05, 0.06)
            self.extrude_height = uniform(0.015, 0.02)
            self.stand_depth_ratio = uniform(0.85, 0.95)
            self.stand_scale = uniform(0.7, 0.85)
            self.bottom_offset = uniform(0.5, 1.5)
            self.back_thickness_ratio = uniform(0, 0.8)
            self.back_size_ratio = uniform(0.55, 0.65)
            self.back_scale = uniform(0.8, 1.0)
            self.seat_thickness_ratio = uniform(0.1, 0.3)
            self.seat_size_ratio = uniform(1.2, 1.6)
            self.has_seat_cut = bool(uniform() < 0.1)
            self.tank_width_ratio = uniform(1.0, 1.2)
            self.tank_size_gap = uniform(0.02, 0.03)
            self.tank_cap_height = uniform(0.03, 0.04)
            self.tank_cap_extrude_amount = uniform(0.005, 0.01)
            self.hardware_radius = uniform(0.015, 0.02)
        self.width = self.size * self.width_ratio
        # NOTE: height_ratio scales self.size but size is resampled on self; effect diluted by normalization, excluded from quartet sampling.
        self.height = self.size * params.height_ratio
        self.curve_scale = self._curve_scale
        self.depth = self.size * self.depth_ratio
        self.stand_depth = self.depth * self.stand_depth_ratio
        self.back_thickness = self.thickness * self.back_thickness_ratio
        self.back_size = self.size * self.back_size_ratio
        self.seat_thickness = self.seat_thickness_ratio * self.thickness
        self.seat_size = self.thickness * self.seat_size_ratio
        self.tank_width = self.width * self.tank_width_ratio
        self.tank_height = self.height * params.tank_height_ratio
        self.tank_size = self.back_size - self.seat_size - self.tank_size_gap
        self.tank_cap_extrude = (
            0
            if params.tank_cap_extrude
            else self.tank_cap_extrude_amount
        )
        self.cover_rotation = -params.cover_rotation
        self.hardware_type = params.flush_control
        self.hardware_cap = params.hardware_cap
        self.hardware_length = params.hardware_length
        self.hardware_on_side = params.hardware_on_side
        self.surface = self._surface
        self.hardware_surface = self._hardware_surface
        self.scratch = self._scratch
        self.edge_wear = self._edge_wear
        self._use_fixed_spawn_draws = spawn_scope

    @property
    def mid_offset(self):
        return (1 - self.size_mid) * self.size

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        return new_bbox(
            -self.mid_offset - self.back_size - self.tank_cap_extrude,
            self.size_mid * self.size + self.thickness + self.thickness,
            -self.width / 2 - self.thickness * 1.1,
            self.width / 2 + self.thickness * 1.1,
            -self.height,
            max(
                self.tank_height,
                -np.sin(self.cover_rotation)
                * (self.seat_size + self.size + self.thickness + self.thickness),
            ),
        )

    def create_asset(self, **params) -> bpy.types.Object:
        upper = self.build_curve()
        lower = deep_clone_obj(upper)
        lower.scale = [self.tube_scale * self.stand_scale] * 3
        lower.location = 0, self.tube_scale * self.mid_offset / 2, -self.depth
        butil.apply_transform(lower, True)
        bottom = deep_clone_obj(upper)
        bottom.scale = [self.stand_scale] * 3
        bottom.location = (
            0,
            self.bottom_offset * self.stand_scale * self.mid_offset,
            -self.height,
        )
        butil.apply_transform(bottom, True)

        obj = self.make_tube(lower, upper)
        seat, cover = self.make_seat(obj)
        stand = self.make_stand(obj, bottom)
        back = self.make_back(obj)
        tank = self.make_tank()
        butil.modify_mesh(obj, "BEVEL", segments=2)
        match self.hardware_type:
            case "button":
                hardware = self.add_button()
            case _:
                hardware = self.add_handle()
        write_attribute(hardware, 1, "hardware", "FACE")
        obj = join_objects([obj, seat, cover, stand, back, tank, hardware])
        obj.rotation_euler[-1] = np.pi / 2
        butil.apply_transform(obj)
        return obj

    def build_curve(self):
        half_width = max(self.width, self.tank_width) / 2
        x_anchors = [0, half_width, 0]
        y_anchors = [-self.size_mid * self.size, 0, self.mid_offset]
        axes = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([1, 0, 0])]
        obj = align_bezier([x_anchors, y_anchors, 0], axes, self.curve_scale)
        butil.modify_mesh(obj, "MIRROR", use_axis=(True, False, False))
        return obj

    def make_tube(self, lower, upper):
        obj = join_objects([upper, lower])
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_mode(type="EDGE")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.bridge_edge_loops(
                number_cuts=64,
                profile_shape_factor=uniform(0.1, 0.2),
                interpolation="SURFACE",
            )
        butil.modify_mesh(
            obj,
            "SOLIDIFY",
            thickness=self.thickness,
            offset=1,
            solidify_mode="NON_MANIFOLD",
            nonmanifold_boundary_mode="FLAT",
        )
        normal = read_normal(obj)
        select_faces(obj, normal[:, -1] > 0.9)
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.extrude_region_move(
                TRANSFORM_OT_translate={
                    "value": (0, 0, self.thickness + self.extrude_height)
                }
            )
        x, y, z = read_co(obj).T
        write_co(
            obj,
            np.stack(
                [x, y, np.clip(z, None, self.thickness + self.extrude_height)],
                -1,
            ),
        )
        return obj

    def make_seat(self, obj):
        seat = self.make_plane(obj)
        cover = deep_clone_obj(seat)
        butil.modify_mesh(seat, "SOLIDIFY", thickness=self.extrude_height, offset=1)
        if self.has_seat_cut:
            cutter = new_cube()
            cutter.scale = [self.thickness] * 3
            cutter.location = 0, -self.thickness / 2 - self.size_mid * self.size, 0
            butil.apply_transform(cutter, True)
            butil.select_none()
            butil.modify_mesh(seat, "BOOLEAN", object=cutter, operation="DIFFERENCE")
            butil.delete(cutter)
        butil.modify_mesh(seat, "BEVEL", segments=2)

        x, y, _ = read_edge_center(cover).T
        i = np.argmin(np.abs(x) + np.abs(y))
        selection = np.full(len(x), False)
        selection[i] = True
        select_edges(cover, selection)
        with butil.ViewportMode(cover, "EDIT"):
            bpy.ops.mesh.loop_multi_select()
            bpy.ops.mesh.fill_grid()
        butil.modify_mesh(cover, "SOLIDIFY", thickness=self.extrude_height, offset=1)
        cover.location = [
            0,
            -self.mid_offset - self.seat_size + self.extrude_height / 2,
            -self.extrude_height / 2,
        ]
        butil.apply_transform(cover, True)
        cover.rotation_euler[0] = self.cover_rotation
        cover.location = [
            0,
            self.mid_offset + self.seat_size - self.extrude_height / 2,
            self.extrude_height * 1.5,
        ]
        butil.apply_transform(cover, True)
        butil.modify_mesh(cover, "BEVEL", segments=2)
        return seat, cover

    def make_plane(self, obj):
        select_faces(obj, lambda x, y, z: z > self.extrude_height * 2 / 3)
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.duplicate_move()
            bpy.ops.mesh.separate(type="SELECTED")
        seat = next(o for o in bpy.context.selected_objects if o != obj)
        butil.select_none()
        select_vertices(seat, lambda x, y, z: y > self.mid_offset + self.seat_thickness)
        with butil.ViewportMode(seat, "EDIT"):
            bpy.ops.mesh.extrude_edges_move(
                TRANSFORM_OT_translate={
                    "value": (0, self.seat_size + self.thickness * 2, 0)
                }
            )
        x, y, z = read_co(seat).T
        write_co(
            seat,
            np.stack([x, np.clip(y, None, self.mid_offset + self.seat_size), z], -1),
        )
        return seat

    def make_stand(self, obj, bottom):
        co = read_co(obj)[read_edges(obj).reshape(-1)].reshape(-1, 2, 3)
        horizontal = np.abs(normalize(co[:, 0] - co[:, 1])[:, -1]) < 0.1
        x, y, z = read_edge_center(obj).T
        under_depth = z < -self.stand_depth
        i = np.argmin(y - horizontal - under_depth)
        selection = np.full(len(co), False)
        selection[i] = True
        select_edges(obj, selection)
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.loop_multi_select()
            bpy.ops.mesh.duplicate_move()
            bpy.ops.mesh.separate(type="SELECTED")
        stand = next(o for o in bpy.context.selected_objects if o != obj)
        stand = join_objects([stand, bottom])
        depth_scale = 1.0 + self.stand_depth / max(self.depth, 1e-6)
        stand.scale = [self.stand_scale, self.stand_scale, depth_scale]
        butil.apply_transform(stand, True)
        with butil.ViewportMode(stand, "EDIT"):
            bpy.ops.mesh.select_mode(type="EDGE")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.bridge_edge_loops(
                number_cuts=64,
                profile_shape_factor=uniform(0.0, 0.15),
            )
        return stand

    def make_back(self, obj):
        back = read_center(obj)[:, 1] > self.mid_offset - self.back_thickness
        back_facing = read_normal(obj)[:, 1] > 0.1
        butil.select_none()
        select_faces(obj, back & back_facing)
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.region_to_loop()
            bpy.ops.mesh.duplicate_move()
            bpy.ops.mesh.separate(type="SELECTED")
        back = next(o for o in bpy.context.selected_objects if o != obj)
        butil.modify_mesh(back, "CORRECTIVE_SMOOTH")
        butil.select_none()
        with butil.ViewportMode(back, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.extrude_edges_move(
                TRANSFORM_OT_translate={
                    "value": (0, self.back_size + self.thickness * 2, 0)
                }
            )
            bpy.ops.transform.resize(value=(self.back_scale, self.back_scale, 1))
            bpy.ops.mesh.edge_face_add()
        back.location[1] -= 0.01
        butil.apply_transform(back, True)
        x, y, z = read_co(back).T
        write_co(
            back,
            np.stack([x, np.clip(y, None, self.mid_offset + self.back_size), z], -1),
        )
        return back

    def make_tank(self):
        tank = new_cube()
        tank.scale = self.tank_width / 2, self.tank_size / 2, self.tank_height / 2
        tank.location = (
            0,
            self.mid_offset + self.back_size - self.tank_size / 2,
            self.tank_height / 2,
        )
        butil.apply_transform(tank, True)
        subsurf(tank, 2, True)
        butil.modify_mesh(tank, "BEVEL", segments=2)
        cap = new_cube()
        cap.scale = (
            self.tank_width / 2 + self.tank_cap_extrude,
            self.tank_size / 2 + self.tank_cap_extrude,
            self.tank_cap_height / 2,
        )
        cap.location = (
            0,
            self.mid_offset + self.back_size - self.tank_size / 2,
            self.tank_height,
        )
        butil.apply_transform(cap, True)
        butil.modify_mesh(
            cap, "BEVEL", width=uniform(0, self.extrude_height), segments=4
        )
        tank = join_objects([tank, cap])
        return tank

    def add_button(self):
        obj = new_cylinder()
        obj.scale = (
            self.hardware_radius,
            self.hardware_radius,
            self.tank_cap_height / 2 + self.hardware_radius,
        )
        obj.location = (
            0,
            self.mid_offset + self.back_size - self.tank_size / 2,
            self.tank_height,
        )
        butil.apply_transform(obj, True)
        return obj

    def add_handle(self):
        obj = new_cylinder()
        obj.scale = self.hardware_radius, self.hardware_radius, self.hardware_cap
        obj.rotation_euler[0] = np.pi / 2
        butil.apply_transform(obj, True)
        lever = new_cylinder()
        lever.scale = (
            self.hardware_radius / 2,
            self.hardware_radius / 2,
            self.hardware_length,
        )
        lever.rotation_euler[1] = np.pi / 2
        lever.location = [
            -self.hardware_radius * uniform(0, 0.5),
            -self.hardware_cap,
            -self.hardware_radius * uniform(0, 0.5),
        ]
        butil.apply_transform(lever, True)
        obj = join_objects([obj, lever])
        if self.hardware_on_side:
            obj.location = [
                -self.tank_width / 2 + self.hardware_radius + uniform(0.01, 0.02),
                self.mid_offset + self.back_size - self.tank_size,
                self.tank_height - self.hardware_radius - uniform(0.02, 0.03),
            ]
        else:
            obj.location = [
                -self.tank_width / 2,
                self.mid_offset
                + self.back_size
                - self.tank_size
                + self.hardware_radius
                + uniform(0.01, 0.02),
                self.tank_height - self.hardware_radius - uniform(0.02, 0.03),
            ]
            obj.rotation_euler[-1] = -np.pi / 2
        butil.apply_transform(obj, True)
        butil.modify_mesh(obj, "BEVEL", width=uniform(0.005, 0.01), segments=2)
        return obj

    def finalize_assets(self, assets):
        self.surface.apply(assets, clear=True, metal_color="plain")
        self.hardware_surface.apply(assets, "hardware", metal_color="natural")
        if self.scratch:
            self.scratch.apply(assets)
        if self.edge_wear:
            self.edge_wear.apply(assets)
