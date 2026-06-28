# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from __future__ import annotations

from typing import Annotated, ClassVar

import bmesh
import bpy
import numpy as np
from numpy.random import uniform
from pydantic import Field

from infinigen.assets.composition import material_assignments
from infinigen.assets.objects.bathroom.bathtub import BathtubFactory
from infinigen.assets.objects.table_decorations import TapFactory
from infinigen.assets.utils.autobevel import BevelSharp
from infinigen.assets.utils.decorate import read_co, subdivide_edge_ring, subsurf
from infinigen.assets.utils.object import (
    join_objects,
    new_base_cylinder,
    new_bbox,
    new_cube,
)
from infinigen.core import surface
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import (
    AssetParameters,
)
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform, weighted_sample


class BathroomSinkParameters(AssetParameters):
    width: Annotated[float, Field(ge=0.6, le=0.9, json_schema_extra={"editable": True})]
    size_ratio: Annotated[float, Field(ge=0.55, le=0.8, json_schema_extra={"editable": True})]
    depth_ratio: Annotated[float, Field(ge=0.2, le=0.4, json_schema_extra={"editable": True})]
    has_curve: Annotated[
        bool, Field(json_schema_extra={"editable": True, "kind": "bool"})
    ] = False
    has_legs: Annotated[
        bool, Field(json_schema_extra={"editable": False, "kind": "bool"})
    ] = False
    is_stand_circular: Annotated[
        bool, Field(json_schema_extra={"editable": True, "kind": "bool"})
    ] = False


def _init_bathroom_sink_state(
    inst: BathroomSinkFactory, seed: int, params: BathroomSinkParameters
) -> None:
    depth = params.width * params.depth_ratio
    with FixedSeed(seed):
        prob = np.array([2, 2])
        inst.bathtub_type = np.random.choice(
            ["alcove", "freestanding"], p=prob / prob.sum()
        )
        inst.contour_fn = inst.make_box_contour
        disp_x0 = uniform(0, 0.2)
        inst.disp_y = uniform(0, 0.1)
        inst.has_curve = uniform() < 0.5
        inst.has_legs = uniform() < 0.5
        inst.leg_height = uniform(0.2, 0.3) * depth
        inst.leg_side = uniform(0.05, 0.1)
        inst.leg_radius = uniform(0.02, 0.03)
        inst.leg_y_scale = uniform()
        inst.leg_subsurf_level = int(np.random.randint(0, 3))
        inst.taper_factor = uniform(-0.1, 0.1)
        inst.stretch_factor = uniform(-0.2, 0.2)
        inst.hole_radius = uniform(0.015, 0.02)
        scratch_draw = uniform()
        edge_wear_draw = uniform()
        inst.leg_bevel_factor = uniform(0.3, 0.7)
        inst.freestanding_z_factor = uniform(0.5, 0.7)
        inst.hole_x_ratio = uniform(0.35, 0.4)
        inst.alcove_levels = np.random.randint(1, 3) if inst.has_base else 1
        inst.thickness = (
            uniform(0.04, 0.08) if inst.has_base else uniform(0.02, 0.04)
        )
        inst.leg_surface_material_gen = weighted_sample(
            material_assignments.metal_neutral
        )
        inst.hole_surface_material_gen = weighted_sample(
            material_assignments.metal_neutral
        )
        inst.beveler = BevelSharp(mult=5, segments=5)
        inst.levels = 5
        inst.side_levels = 2
        inst.is_hole_centered = True
        inst.sink_types = np.random.choice(["undermount", "drop-in", "vessel"])
        inst.has_stand = False
        match inst.sink_types:
            case "undermount":
                inst.bathtub_type = "freestanding"
                inst.has_extrude = uniform() < 0.7
            case "drop-in":
                inst.bathtub_type = "alcove"
                inst.has_extrude = True
            case _:
                inst.bathtub_type = np.random.choice(["alcove", "freestanding"])
                inst.has_extrude = uniform() < 0.7
                inst.has_stand = True
        inst.tap_factory = TapFactory(inst.factory_seed)
        inst.alcove_levels = 0 if uniform() < 0.5 else int(np.random.randint(2, 4))
        inst.thickness = 0.01 if inst.has_base else uniform(0.01, 0.03)
        inst.size_extrude = uniform(0.2, 0.35)
        inst.tap_offset = uniform(0.0, 0.05)
        inst.stand_radius = params.width / 2 * log_uniform(0.15, 0.2)
        inst.stand_bottom = (
            params.width * log_uniform(0.2, 0.3)
            if uniform() < 0.6
            else inst.stand_radius
        )
        inst.stand_height = uniform(0.7, 0.9) - depth
        inst.is_stand_circular = params.is_stand_circular
        surface_gen_class = weighted_sample(material_assignments.bathroom_touchsurface)
        inst.surface_material_gen = surface_gen_class()
    scratch_prob, edge_wear_prob = material_assignments.wear_tear_prob
    scratch_fn, edge_wear_fn = material_assignments.wear_tear
    inst.scratch = None if scratch_draw > scratch_prob else scratch_fn()
    inst.edge_wear = None if edge_wear_draw > edge_wear_prob else edge_wear_fn()
    inst.width = params.width
    inst.size = params.width * params.size_ratio
    inst.depth = depth
    inst.disp_x = np.array([disp_x0, disp_x0])


class BathroomSinkFactory(BathtubFactory):
    parameters_model: ClassVar[type[AssetParameters]] = BathroomSinkParameters

    def __init__(self, factory_seed, coarse=False):
        AssetFactory.__init__(self, factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> BathroomSinkParameters:
        return BathroomSinkParameters(
            seed=seed,
            width=uniform(0.6, 0.9),
            size_ratio=log_uniform(0.55, 0.8),
            depth_ratio=log_uniform(0.2, 0.4),
            has_curve=bool(uniform() < 0.5),
            has_legs=bool(uniform() < 0.5),
            is_stand_circular=bool(uniform() < 0.5),
        )

    def apply_parameters(
        self, params: BathroomSinkParameters, *, spawn_scope: bool = True
    ) -> None:
        _init_bathroom_sink_state(self, params.seed, params)
        self.has_curve = params.has_curve
        self.has_legs = params.has_legs
        self.is_stand_circular = params.is_stand_circular
        self._use_fixed_spawn_draws = spawn_scope

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        return new_bbox(
            -(self.size_extrude + 1) * self.size,
            0,
            0,
            self.width,
            -self.stand_height if self.has_stand else 0,
            self.depth,
        )

    def create_asset(self, **params) -> bpy.types.Object:
        self.surface = self.surface_material_gen()
        if self.has_base:
            obj = self.make_base()
            cutter = self.make_cutter()
            butil.modify_mesh(obj, "BOOLEAN", object=cutter, operation="DIFFERENCE")
            butil.delete(cutter)
        else:
            obj = self.make_bowl()
            self.remove_top(obj)
            butil.modify_mesh(obj, "SOLIDIFY", thickness=self.thickness)
            subsurf(obj, self.side_levels)
        obj.location = np.array(obj.location) - np.min(read_co(obj), 0)
        butil.apply_transform(obj, True)
        obj.scale = np.array([self.width, self.size, self.depth]) / np.array(
            obj.dimensions
        )
        butil.apply_transform(obj, True)
        if self.has_extrude:
            self.extrude_back(obj)
        if self.has_stand:
            self.add_stand(obj)
        hole = self.add_hole(obj)
        obj = join_objects([obj, hole])
        obj.rotation_euler[-1] = np.pi / 2
        butil.apply_transform(obj, True)
        surface.assign_material(obj, self.surface)
        if self.has_extrude:
            tap = self.tap_factory(np.random.randint(1e7))
            min_x = np.min(read_co(tap)[:, 0])
            tap.location = (
                (-1 - self.size_extrude + self.tap_offset) * self.size - min_x,
                self.width / 2,
                self.depth,
            )
            butil.apply_transform(tap, True)
            obj = join_objects([obj, tap])
        return obj

    def extrude_back(self, obj):
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_mode(type="FACE")
            bpy.ops.mesh.select_all(action="DESELECT")
            bm = bmesh.from_edit_mesh(obj.data)
            for f in bm.faces:
                f.select_set(
                    f.calc_center_median()[1] > self.size / 2 and f.normal[1] > 0.1
                )
            bm.select_flush(False)
            bmesh.update_edit_mesh(obj.data)
            bpy.ops.mesh.extrude_region_move(
                TRANSFORM_OT_translate={"value": (0, self.size_extrude * self.size, 0)}
            )

    def add_stand(self, obj):
        if self.is_stand_circular:
            stand = new_base_cylinder(vertices=16)
            stand.scale = self.stand_radius, self.stand_radius, self.stand_height / 2
        else:
            stand = new_cube()
            stand.scale = (
                self.stand_radius * 1.5,
                self.stand_radius * 1.5,
                self.stand_height / 2,
            )
        stand.location = self.width / 2, self.size / 2, -self.stand_height / 2
        butil.apply_transform(stand, True)
        subdivide_edge_ring(stand, np.random.randint(3, 6))
        with butil.ViewportMode(stand, "EDIT"):
            bpy.ops.mesh.select_mode(type="FACE")
            bm = bmesh.from_edit_mesh(stand.data)
            for f in bm.faces:
                f.select_set(f.normal[-1] < -0.1)
            bm.select_flush(False)
            bmesh.update_edit_mesh(stand.data)
            bpy.ops.transform.resize(
                value=(
                    self.stand_bottom / self.stand_radius,
                    self.stand_bottom / self.stand_radius,
                    1,
                )
            )
        subsurf(stand, 2, True)
        subsurf(stand, 1)
        obj = join_objects([obj, stand])
        return obj

    def finalize_assets(self, assets):
        if self.scratch:
            self.scratch.apply(assets)
        if self.edge_wear:
            self.edge_wear.apply(assets)


class StandingSinkParameters(BathroomSinkParameters):
    pass


class StandingSinkFactory(BathroomSinkFactory):
    parameters_model: ClassVar[type[AssetParameters]] = StandingSinkParameters

    def __init__(self, factory_seed, coarse=False):
        AssetFactory.__init__(self, factory_seed, coarse)
        self.init_legacy_parameters()

    def apply_parameters(
        self, params: StandingSinkParameters, *, spawn_scope: bool = True
    ) -> None:
        _init_bathroom_sink_state(self, params.seed, params)
        self.bathtub_type = "freestanding"
        self.has_extrude = True
        self.has_stand = True
        self.has_curve = params.has_curve
        self.has_legs = params.has_legs
        self.is_stand_circular = params.is_stand_circular
        self._use_fixed_spawn_draws = spawn_scope
