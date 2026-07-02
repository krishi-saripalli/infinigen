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

from .base import apply_tableware_from_draws, sample_tableware_base
from .pan import PanFactory, PanParameters

_POT_TIER1_FIELDS = (
    "bar_radius",
    "grid_offset",
    "guard_depth_mult",
    "has_guard",
    "has_handle_hole",
    "hole_location_frac",
    "hole_scale",
    "lower_thresh",
    "n_vertices",
    "pot_guard_depth_mult",
    "r_mid",
    "s_handle",
    "scale",
    "thickness",
    "x_guard_extra",
    "x_handle",
    "z_handle_frac",
    "z_handle_mid_frac",
)


class PotParameters(PanParameters):
    pot_depth: Annotated[float, Field(ge=0.6, le=2.0, json_schema_extra={"editable": True})]
    has_bar: Annotated[
        bool, Field(json_schema_extra={"editable": False, "kind": "bool"})
    ] = True
    bar_scale_x: Annotated[float, Field(ge=0.6, le=1.0, json_schema_extra={"editable": True})]


for _field in _POT_TIER1_FIELDS:
    PotParameters.model_fields.pop(_field, None)
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
            if k not in _POT_TIER1_FIELDS
        }
        scale_mid = log_uniform(0.6, 1.5)
        return PotParameters(
            **pan_data,
            pot_depth=log_uniform(0.6, 2.0),
            has_bar=True,
            bar_scale_x=np.clip(log_uniform(0.6, 1.0) * scale_mid, 0.6, 1.0),
        )

    def _sample_spawn_parameters(
        self, params: PotParameters, seed: int, i: int
    ) -> PotParameters:
        return params

    def apply_parameters(
        self, params: PotParameters, *, spawn_scope: bool = True
    ) -> None:
        with FixedSeed(params.seed):
            pan_scale = log_uniform(0.1, 0.15)
            base = sample_tableware_base(params.seed)
            lower_thresh = base["lower_thresh"]
            self.has_handle_hole = bool(uniform() < 0.6)
            self.x_handle = log_uniform(1.2, 2.0)
            z_handle_frac = uniform(0, 0.2)
            self.z_handle = self.x_handle * z_handle_frac
            self.z_handle_mid = uniform(0.6, 0.8) * self.z_handle
            self.s_handle = log_uniform(0.8, 1.2)
            x_guard_extra = uniform(0, 0.2)
            self.x_guard = params.r_expand + x_guard_extra * self.x_handle
            self.bar_radius = log_uniform(0.2, 0.3)
            pot_guard_depth_mult = log_uniform(0.5, 1.0)
            self.r_mid = log_uniform(1.0, 1.3)
            self.thickness = log_uniform(0.04, 0.06)
            bar_height_frac = uniform(0.75, 0.85)
            bar_inner_radius_ratio = log_uniform(0.2, 0.4)
            bar_scale_z = np.clip(log_uniform(0.6, 1.2) * log_uniform(0.6, 1.5), 0.6, 1.2)
            bar_taper = log_uniform(0.3, 0.8)
            bar_y_rotation = uniform(-np.pi / 6, 0)
            bar_x_offset_frac = uniform(-0.1, 0.1)
            self.bar_x = 1 + uniform(-self.bar_radius, self.bar_radius) * 0.05
            n_factor = log_uniform(4, 8)
            n = 4 * int(n_factor)
            grid_offset = int(np.random.randint(n // 4))
            hole_scale = uniform(0.06, 0.1)
            hole_location_frac = uniform(0.8, 0.9)
        apply_tableware_from_draws(
            self,
            seed=params.seed,
            lower_thresh=lower_thresh,
            scale=pan_scale,
            metal_color=None,
        )
        self.depth = params.depth
        self.r_expand = params.r_expand
        self.scale = pan_scale
        self.has_bar = params.has_bar
        self.has_handle = not self.has_bar
        self.has_guard = not self.has_bar
        # pot_depth's range (0.6-2.0) regularly exceeds depth's range
        # (0.3-0.8), so an unclamped bar_height often sits above the rim
        # entirely -- the boolean-subtracted handle ring never overlaps the
        # body and renders as a fully detached floating ring/bar. Clamp so
        # the handle always at least reaches the rim.
        self.bar_height = min(params.pot_depth * bar_height_frac, self.depth)
        self.bar_inner_radius = self.bar_radius * bar_inner_radius_ratio
        self.bar_scale = (params.bar_scale_x, 1.0, bar_scale_z)
        self.bar_taper = bar_taper
        self.bar_y_rotation = bar_y_rotation
        self.bar_x_offset = self.bar_radius * bar_x_offset_frac
        self.guard_type = "round"
        self.guard_depth = pot_guard_depth_mult * self.thickness
        self._use_fixed_spawn_draws = spawn_scope
        if spawn_scope:
            self._n = n
            self._grid_offset = grid_offset
            self._hole_scale = hole_scale
            self._hole_location_frac = hole_location_frac

    def post_init(self) -> None:
        if not hasattr(self, "has_bar"):
            return
        self.has_handle = not self.has_bar
        self.has_guard = not self.has_bar
        if not self.has_bar or getattr(self, "_use_fixed_spawn_draws", False):
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
