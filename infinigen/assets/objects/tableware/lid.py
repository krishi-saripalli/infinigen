# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from __future__ import annotations

from typing import Annotated, ClassVar

import bpy
import numpy as np
from numpy.random import uniform
from pydantic import Field

from infinigen.assets.composition import material_assignments
from infinigen.assets.utils.decorate import read_center, subsurf, write_co
from infinigen.assets.utils.draw import spin
from infinigen.assets.utils.object import join_objects, new_cylinder, new_line
from infinigen.core import surface
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import (
    AssetParameters,
    ParameterizedAssetFactory,
)
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import weighted_sample


class LidParameters(AssetParameters):
    x_length: Annotated[float, Field(ge=0.08, le=0.15, json_schema_extra={"editable": False})]
    z_height_ratio: Annotated[
        float, Field(ge=0.0, le=0.5, json_schema_extra={"editable": True})
    ]
    is_glass: Annotated[
        bool, Field(json_schema_extra={"editable": False, "kind": "bool"})
    ] = True
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
    # NOTE: only affects handle path when handle_type=handle (sampled at init).
    handle_height_ratio: Annotated[
        float, Field(ge=0.1, le=0.25, json_schema_extra={"editable": False})
    ] = 0.15
    handle_type: Annotated[
        str,
        Field(
            json_schema_extra={
                "editable": False,
                "kind": "enum",
                "choices": ["handle", "knob"],
            }
        ),
    ] = "handle"
    z_anchor_frac: Annotated[
        float, Field(ge=0.7, le=0.8, json_schema_extra={"editable": True})
    ] = 0.75
    handle_subsurf_level: Annotated[
        int, Field(ge=0, le=2, json_schema_extra={"editable": True})
    ] = 1

class LidFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = LidParameters

    def __init__(self, factory_seed, coarse=False):
        super(LidFactory, self).__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_materials(self, seed: int, is_glass: bool) -> None:
        with FixedSeed(seed):
            if is_glass:
                surface_gen_class = weighted_sample(
                    material_assignments.appliance_front_maybeglass
                )
            else:
                surface_gen_class = weighted_sample(material_assignments.decorative_hard)
            self.surface_material_gen = surface_gen_class()
            rim_surface_gen_class = weighted_sample(material_assignments.metals)
            self.rim_surface_material_gen = rim_surface_gen_class()
            handle_surface_gen_class = weighted_sample(material_assignments.decorative_hard)
            self.handle_surface_material_gen = handle_surface_gen_class()

    def _sample_init_parameters(self, seed: int) -> LidParameters:
        x_length = uniform(0.08, 0.15)
        is_glass = True
        self._sample_materials(seed, True)
        handle_type = "handle"
        return LidParameters(
            seed=seed,
            x_length=x_length,
            z_height_ratio=uniform(0, 0.5),
            is_glass=is_glass,
            scratch_draw=uniform(),
            edge_wear_draw=uniform(),
            handle_type=handle_type,
            handle_height_ratio=uniform(0.1, 0.15),
            z_anchor_frac=uniform(0.7, 0.8),
            handle_subsurf_level=int(np.random.randint(0, 3)),
        )

    def _sample_spawn_parameters(
        self, params: LidParameters, seed: int, i: int
    ) -> LidParameters:
        return params.model_copy(
            update={
                "z_anchor_frac": uniform(0.7, 0.8),
                "handle_subsurf_level": int(np.random.randint(0, 3)),
            }
        )

    def apply_parameters(
        self, params: LidParameters, *, spawn_scope: bool = True
    ) -> None:
        is_glass = params.is_glass
        self._sample_materials(params.seed, is_glass)
        self.handle_type = params.handle_type
        scratch_prob, edge_wear_prob = material_assignments.wear_tear_prob
        scratch_fn, edge_wear_fn = material_assignments.wear_tear
        self.scratch = None if params.scratch_draw > scratch_prob else scratch_fn()
        self.edge_wear = None if params.edge_wear_draw > edge_wear_prob else edge_wear_fn()
        self.x_length = params.x_length
        self.z_height = params.x_length * params.z_height_ratio
        # NOTE: thickness sampled on self from seed; excluded from quartet sampling (uniform scale normalized away in point clouds).
        with FixedSeed(params.seed):
            self.thickness = uniform(0.003, 0.005)
        self.is_glass = is_glass
        self.hardware_type = None
        # NOTE: rim_height_mult, handle_radius_ratio, and handle_width_ratio do not elicit a clear visual change in exported geometry; excluded from quartet sampling.
        with FixedSeed(params.seed):
            self.rim_height_mult = uniform(1, 2)
            self.handle_radius_ratio = uniform(0.15, 0.25)
            self.handle_width_ratio = uniform(0.25, 0.3)
        self.rim_height = self.rim_height_mult * self.thickness
        self.handle_height = params.x_length * params.handle_height_ratio
        self.handle_radius = params.x_length * self.handle_radius_ratio
        self.handle_width = params.x_length * self.handle_width_ratio
        self._z_anchor_frac = params.z_anchor_frac
        self._handle_subsurf_level = params.handle_subsurf_level
        self._use_fixed_spawn_draws = spawn_scope

    def create_asset(self, **params) -> bpy.types.Object:
        self.surface = self.surface_material_gen()
        self.rim_surface = self.rim_surface_material_gen()
        self.handle_surface = self.handle_surface_material_gen()

        z_mid = (
            self._z_anchor_frac
            if self._use_fixed_spawn_draws
            else uniform(0.7, 0.8)
        )
        x_anchors = 0, 0.01, self.x_length / 2, self.x_length
        z_anchors = self.z_height, self.z_height, self.z_height * z_mid, 0
        obj = spin((x_anchors, 0, z_anchors))
        butil.modify_mesh(obj, "SOLIDIFY", thickness=self.thickness, offset=0)
        butil.modify_mesh(obj, "BEVEL", width=self.thickness / 2, segments=4)

        surface.assign_material(obj, self.surface)
        parts = [obj]
        if self.is_glass:
            parts.append(self.add_rim())
        match self.handle_type:
            case "handle":
                parts.append(self.add_handle(obj))
            case _:
                parts.append(self.add_knob())
        obj = join_objects(parts)
        return obj

    def add_rim(self):
        butil.select_none()
        bpy.ops.mesh.primitive_torus_add(
            major_radius=self.x_length,
            minor_radius=self.thickness / 2,
            major_segments=128,
        )
        obj = bpy.context.active_object
        obj.scale[-1] = self.rim_height / self.thickness
        butil.apply_transform(obj)
        surface.assign_material(obj, self.rim_surface)
        return obj

    def add_handle(self, obj):
        center = read_center(obj)
        i = np.argmin(
            np.abs(center[:, :2] - np.array([self.handle_width, 0])[np.newaxis, :]).sum(
                -1
            )
        )
        z_offset = center[i, -1]
        obj = new_line(3)
        write_co(
            obj,
            np.array(
                [
                    [-self.handle_width, 0, 0],
                    [-self.handle_width, 0, self.handle_height],
                    [self.handle_width, 0, self.handle_height],
                    [self.handle_width, 0, 0],
                ]
            ),
        )
        subsurf(obj, self._handle_subsurf_level)
        butil.select_none()
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_mode(type="EDGE")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.extrude_edges_move(
                TRANSFORM_OT_translate={"value": (0, self.thickness * 2, 0)}
            )
        butil.modify_mesh(obj, "SOLIDIFY", thickness=self.thickness, offset=0)
        butil.modify_mesh(obj, "BEVEL", width=self.thickness / 2, segments=4)
        obj.location = 0, -self.thickness, z_offset
        butil.apply_transform(obj, True)
        surface.assign_material(obj, self.handle_surface)
        return obj

    def add_knob(self):
        obj = new_cylinder()
        obj.scale = *([self.thickness * uniform(1, 2)] * 2), self.handle_height
        obj.location[-1] = self.z_height
        butil.apply_transform(obj, True)
        butil.modify_mesh(obj, "BEVEL", width=self.thickness / 2, segments=4)
        top = new_cylinder()
        top.scale = (
            self.handle_radius,
            self.handle_radius,
            self.thickness * uniform(1, 2),
        )
        top.location[-1] = self.z_height + self.handle_height
        butil.apply_transform(top, True)
        butil.modify_mesh(top, "BEVEL", width=self.thickness / 2, segments=4)
        obj = join_objects([obj, top])
        surface.assign_material(obj, self.handle_surface)
        return obj

    def finalize_assets(self, assets):
        if self.scratch:
            self.scratch.apply(assets)
        if self.edge_wear:
            self.edge_wear.apply(assets)
