# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from __future__ import annotations

from typing import Annotated, ClassVar

import bpy
import numpy as np
from numpy.random import uniform
from pydantic import Field

from infinigen.assets.utils.decorate import read_co, subsurf, write_attribute
from infinigen.assets.utils.object import join_objects, new_bbox
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import AssetParameters
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform

from .base import apply_tableware_from_draws
from .pan import PanFactory, PanParameters


class PotParameters(PanParameters):
    pot_depth: Annotated[float, Field(ge=0.6, le=2.0, json_schema_extra={"editable": True})]
    has_bar: Annotated[
        bool, Field(json_schema_extra={"editable": False, "kind": "bool"})
    ] = True
    bar_radius: Annotated[float, Field(ge=0.2, le=0.3, json_schema_extra={"editable": False})]
    bar_scale_x: Annotated[float, Field(ge=0.6, le=1.0, json_schema_extra={"editable": True})]
    pot_guard_depth_mult: Annotated[
        float, Field(ge=0.5, le=1.0, json_schema_extra={"editable": True})
    ]


PotParameters.model_fields.pop("scale")
PotParameters.model_fields.pop("r_mid")
PotParameters.model_fields.pop("thickness")
PotParameters.model_fields.pop("guard_depth_mult")
PotParameters.model_rebuild(force=True)


class PotFactory(PanFactory):
    parameters_model: ClassVar[type[AssetParameters]] = PotParameters

    def __init__(self, factory_seed, coarse=False):
        AssetFactory.__init__(self, factory_seed, coarse)
        self.has_handle = True
        self.pre_level = 2
        self.guard_type = "round"
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> PotParameters:
        pan_params = PanFactory._sample_init_parameters(self, seed)
        pan_data = {
            k: v
            for k, v in pan_params.model_dump().items()
            if k not in ("scale", "r_mid", "thickness", "guard_depth_mult")
        }
        bar_radius = log_uniform(0.2, 0.3)
        scale_mid = log_uniform(0.6, 1.5)
        return PotParameters(
            **pan_data,
            pot_depth=log_uniform(0.6, 2.0),
            has_bar=True,
            bar_radius=bar_radius,
            bar_scale_x=np.clip(log_uniform(0.6, 1.0) * scale_mid, 0.6, 1.0),
            pot_guard_depth_mult=log_uniform(0.5, 1.0),
        )

    def apply_parameters(
        self, params: PotParameters, *, spawn_scope: bool = True
    ) -> None:
        # NOTE: scale sampled on self from seed; excluded from quartet sampling (uniform scale normalized away in point clouds).
        with FixedSeed(params.seed):
            pan_scale = log_uniform(0.1, 0.15)
        apply_tableware_from_draws(
            self,
            seed=params.seed,
            lower_thresh=params.lower_thresh,
            scale=pan_scale,
            scratch_draw=params.scratch_draw,
            edge_wear_draw=params.edge_wear_draw,
            metal_color=None,
        )
        self.has_handle_hole = params.has_handle_hole
        self.x_handle = params.x_handle
        self.z_handle = params.x_handle * params.z_handle_frac
        self.z_handle_mid = params.z_handle_mid_frac * self.z_handle
        self.s_handle = params.s_handle
        self.x_guard = params.r_expand + params.x_guard_extra * params.x_handle
        self.depth = params.depth
        self.r_expand = params.r_expand
        self.scale = pan_scale
        self.has_bar = params.has_bar
        self.has_handle = not self.has_bar
        # NOTE: has_guard is derived from has_bar and not wired into exported geometry for plant_pot; excluded from quartet sampling.
        self.has_guard = not self.has_bar
        # NOTE: bar_radius only affects geometry when has_bar is True.
        self.bar_radius = params.bar_radius
        # NOTE: r_mid is not wired to exported geometry; excluded from quartet sampling.
        # NOTE: thickness is not wired to exported geometry; excluded from quartet sampling.
        with FixedSeed(params.seed):
            self.r_mid = log_uniform(1.0, 1.3)
            self.thickness = log_uniform(0.04, 0.06)
            # NOTE: bar_height_frac is not wired to exported geometry; excluded from quartet sampling.
            bar_height_frac = uniform(0.75, 0.85)
            # NOTE: bar_inner_radius_ratio is not wired to exported geometry; excluded from quartet sampling.
            bar_inner_radius_ratio = log_uniform(0.2, 0.4)
            # NOTE: bar_scale_z is not wired to exported geometry; excluded from quartet sampling.
            bar_scale_z = np.clip(log_uniform(0.6, 1.2) * log_uniform(0.6, 1.5), 0.6, 1.2)
            # NOTE: bar_taper is not wired to exported geometry; excluded from quartet sampling.
            bar_taper = log_uniform(0.3, 0.8)
            # NOTE: bar_y_rotation is not wired to exported geometry; excluded from quartet sampling.
            bar_y_rotation = uniform(-np.pi / 6, 0)
            # NOTE: bar_x_offset_frac is not wired to exported geometry; excluded from quartet sampling.
            bar_x_offset_frac = uniform(-0.1, 0.1)
            self.bar_x = 1 + uniform(-self.bar_radius, self.bar_radius) * 0.05
        self.bar_height = params.pot_depth * bar_height_frac
        self.bar_inner_radius = self.bar_radius * bar_inner_radius_ratio
        self.bar_scale = (params.bar_scale_x, 1.0, bar_scale_z)
        self.bar_taper = bar_taper
        self.bar_y_rotation = bar_y_rotation
        self.bar_x_offset = self.bar_radius * bar_x_offset_frac
        self.guard_type = "round"
        self.guard_depth = params.pot_guard_depth_mult * self.thickness
        self._use_fixed_spawn_draws = spawn_scope
        if spawn_scope:
            self._n = 4 * int(params.n_vertices)
            self._grid_offset = params.grid_offset
            self._hole_scale = params.hole_scale
            self._hole_location_frac = params.hole_location_frac

    def post_init(self) -> None:
        if not hasattr(self, "has_bar"):
            return
        self.has_handle = not self.has_bar
        self.has_guard = not self.has_bar
        if getattr(self, "_use_fixed_spawn_draws", False):
            return
        self.bar_x = 1 + uniform(-self.bar_radius, self.bar_radius) * 0.05
        self.bar_inner_radius = log_uniform(0.2, 0.4) * self.bar_radius
        self.bar_x_offset = self.bar_radius * uniform(-0.1, 0.1)

    def create_asset(self, **params) -> bpy.types.Object:
        obj = self.make_base()
        if self.has_bar:
            self.add_bar(obj)
        obj.scale = [self.scale] * 3
        butil.apply_transform(obj)
        return obj

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        if self.has_bar:
            radius_ = (
                1
                + self.bar_x_offset
                + self.bar_radius
                + self.bar_inner_radius
                + self.thickness
            )
            obj = new_bbox(
                -radius_,
                radius_,
                -1 - self.thickness,
                1 + self.thickness,
                0,
                self.depth,
            )
        elif self.has_handle:
            obj = new_bbox(
                -1 - self.thickness,
                1 + self.thickness + self.x_handle,
                -1 - self.thickness,
                1 + self.thickness,
                0,
                self.depth,
            )
        else:
            obj = new_bbox(
                -1 - self.thickness,
                1 + self.thickness,
                -1 - self.thickness,
                1 + self.thickness,
                0,
                self.depth,
            )
        obj.scale = (self.scale,) * 3
        butil.apply_transform(obj)
        return obj

    def add_bar(self, obj):
        bars = []
        for side in [-1, 1]:
            bpy.ops.mesh.primitive_torus_add(
                location=(side * (1 + self.bar_x_offset), 0, self.bar_height),
                major_radius=self.bar_radius,
                minor_radius=self.bar_inner_radius,
            )
            bar = bpy.context.active_object
            bar.scale = self.bar_scale
            butil.modify_mesh(
                bar,
                "SIMPLE_DEFORM",
                deform_method="TAPER",
                angle=self.bar_taper,
                deform_axis="X",
            )
            bar.rotation_euler = 0, self.bar_y_rotation, 0 if side == 1 else np.pi
            butil.apply_transform(bar)

            butil.modify_mesh(bar, "BOOLEAN", object=obj, operation="DIFFERENCE")
            butil.select_none()
            objs = butil.split_object(bar)
            i = np.argmax([np.max(read_co(o)[:, 0] * side) for o in objs])
            bar = objs[i]
            objs.remove(bar)
            butil.delete(objs)
            subsurf(bar, 1)
            write_attribute(bar, lambda nw: 1, "guard", "FACE")
            bars.append(bar)
        return join_objects([obj, *bars])
