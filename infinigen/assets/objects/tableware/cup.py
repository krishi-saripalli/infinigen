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
from infinigen.assets.materials import text
from infinigen.assets.objects.tableware.base import (
    TablewareFactory,
    apply_tableware_from_draws,
    sample_tableware_base,
)
from infinigen.assets.utils.decorate import (
    read_co,
    remove_vertices,
    subsurf,
    write_attribute,
)
from infinigen.assets.utils.draw import spin
from infinigen.assets.utils.object import join_objects
from infinigen.assets.utils.uv import wrap_sides
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import (
    AssetParameters,
    ParameterizedAssetFactory,
)
from infinigen.core.util import blender as butil
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform, weighted_sample


class CupParameters(AssetParameters):
    thickness: Annotated[float, Field(ge=0.01, le=0.04, json_schema_extra={"editable": False})]
    # NOTE: only applies when has_guard=True.
    handle_taper_x: Annotated[float, Field(ge=0.0, le=2.0, json_schema_extra={"editable": False})]
    # NOTE: only applies when has_guard=True.
    handle_taper_y: Annotated[float, Field(ge=0.0, le=2.0, json_schema_extra={"editable": False})]
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
    is_short: Annotated[
        bool, Field(json_schema_extra={"editable": False, "kind": "bool"})
    ] = True
    has_guard: Annotated[
        bool, Field(json_schema_extra={"editable": False, "kind": "bool"})
    ] = True
    is_profile_straight: Annotated[
        bool, Field(json_schema_extra={"editable": False, "kind": "bool"})
    ] = True
    handle_type: Annotated[
        str,
        Field(
            json_schema_extra={
                "editable": False,
                "kind": "enum",
                "choices": ["shear", "round"],
            }
        ),
    ] = "shear"
    depth: Annotated[float, Field(ge=0.25, le=1.0, json_schema_extra={"editable": True})]
    # NOTE: anchor layout differs when is_profile_straight=False.
    x_lowest: Annotated[float, Field(ge=0.6, le=1.0, json_schema_extra={"editable": False})]
    # NOTE: only applies when has_guard=True.
    handle_location: Annotated[
        float, Field(ge=-0.1, le=0.65, json_schema_extra={"editable": False})
    ]
    handle_radius_ratio: Annotated[
        float, Field(ge=0.2, le=0.4, json_schema_extra={"editable": False})
    ]
    handle_angle: Annotated[
        float, Field(ge=0.009733, le=0.544872, json_schema_extra={"editable": True})
    ] = 0.2


class CupFactory(ParameterizedAssetFactory, TablewareFactory):
    parameters_model: ClassVar[type[AssetParameters]] = CupParameters
    allow_transparent = True

    def __init__(self, factory_seed, coarse=False):
        AssetFactory.__init__(self, factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_wrap_surface(self, seed: int) -> None:
        with FixedSeed(seed):
            wrap_surface = weighted_sample(material_assignments.graphicdesign)()()
            if wrap_surface == text.Text:
                wrap_surface = text.Text(seed, False)
            self.wrap_surface = wrap_surface

    def _apply_cup_branches(self, params: CupParameters) -> None:
        self.is_short = params.is_short
        self.is_profile_straight = params.is_profile_straight
        self.has_guard = params.has_guard
        self.handle_type = params.handle_type
        self.x_end = 0.25
        self.x_lowest = params.x_lowest
        self.depth = params.depth
        self.handle_location = params.handle_location
        self.handle_radius = params.depth * params.handle_radius_ratio
        # NOTE: handle_inner_radius_ratio does not elicit a clear visual change in exported geometry; excluded from quartet sampling.
        with FixedSeed(params.seed):
            self.handle_inner_radius = self.handle_radius * log_uniform(0.2, 0.3)
        self.has_wrap = True

    def _sample_init_parameters(self, seed: int) -> CupParameters:
        base = sample_tableware_base(seed)
        self._init_tableware_base()
        is_short = True
        with FixedSeed(seed):
            if is_short:
                is_profile_straight = bool(uniform(0, 1) < 0.2)
                has_guard = bool(uniform(0, 1) < 0.8)
            else:
                is_profile_straight = True
                has_guard = False
            handle_type = "shear" if uniform(0, 1) < 0.5 else "round"
        if is_short:
            x_lowest = log_uniform(0.6, 0.9)
            depth = log_uniform(0.25, 0.5)
            handle_location = (
                uniform(0.45, 0.65) if uniform(0, 1) < 0.2 else uniform(-0.1, 0.3)
            )
        else:
            x_lowest = log_uniform(0.9, 1.0)
            depth = log_uniform(0.5, 1.0)
            handle_location = uniform(0.45, 0.65)
        self._sample_wrap_surface(seed)
        return CupParameters(
            seed=seed,
            thickness=log_uniform(0.01, 0.04),
            handle_taper_x=uniform(0, 2),
            handle_taper_y=uniform(0, 2),
            scratch_draw=base["scratch_draw"],
            edge_wear_draw=base["edge_wear_draw"],
            is_short=is_short,
            has_guard=has_guard,
            is_profile_straight=is_profile_straight,
            handle_type=handle_type,
            depth=depth,
            x_lowest=x_lowest,
            handle_location=handle_location,
            handle_radius_ratio=uniform(0.2, 0.4),
            handle_angle=uniform(0.009733, 0.544872),
        )

    def _sample_spawn_field_updates(self) -> dict[str, float]:
        return {
            "handle_angle": uniform(0.009733, 0.544872),
        }

    def _sample_spawn_parameters(
        self, params: CupParameters, seed: int, i: int
    ) -> CupParameters:
        return params.model_copy(update=self._sample_spawn_field_updates())

    def apply_parameters(
        self, params: CupParameters, *, spawn_scope: bool = True
    ) -> None:
        # NOTE: scale sampled on self from seed; excluded from quartet sampling (uniform scale normalized away in point clouds).
        with FixedSeed(params.seed):
            self.scale = log_uniform(0.15, 0.3)
            base = sample_tableware_base(params.seed)
            self._lower_thresh = base["lower_thresh"]
            self.x_lower_ratio = log_uniform(0.8, 1.0)
            self.has_inside = bool(uniform(0, 1) < 0.5)
            self._bevel_width_pct = uniform(10, 50)
        apply_tableware_from_draws(
            self,
            seed=params.seed,
            lower_thresh=self._lower_thresh,
            scale=self.scale,
            scratch_draw=params.scratch_draw,
            edge_wear_draw=params.edge_wear_draw,
            has_inside=self.has_inside,
        )
        self._sample_wrap_surface(params.seed)
        self._apply_cup_branches(params)
        self.thickness = params.thickness
        # NOTE: wrap_margin does not elicit a clear visual change in exported geometry; excluded from quartet sampling.
        with FixedSeed(params.seed):
            self.wrap_margin = uniform(0.1, 0.2)
        self.handle_taper_x = params.handle_taper_x
        self.handle_taper_y = params.handle_taper_y
        with FixedSeed(params.seed):
            self._wrap_margin_jitter = uniform(0.0, 0.1)
        self._use_fixed_spawn_draws = spawn_scope
        if spawn_scope:
            self._handle_angle = params.handle_angle

    def create_asset(self, **params) -> bpy.types.Object:
        if self.is_profile_straight:
            x_anchors = 0, self.x_lowest * self.x_end, self.x_end
            z_anchors = 0, 0, self.depth
        else:
            x_anchors = (
                0,
                self.x_lowest * self.x_end,
                (self.x_lowest + self.x_lower_ratio * (1 - self.x_lowest)) * self.x_end,
                self.x_end,
            )
            z_anchors = 0, 0, self.depth * 0.5, self.depth
        anchors = np.array(x_anchors) * self.scale, 0, np.array(z_anchors) * self.scale
        obj = spin(anchors, [1])
        obj.scale = [1 / self.scale] * 3
        butil.apply_transform(obj, True)
        butil.modify_mesh(
            obj,
            "BEVEL",
            True,
            offset_type="PERCENT",
            width_pct=(
                self._bevel_width_pct
                if self._use_fixed_spawn_draws
                else uniform(10, 50)
            ),
            segments=8,
        )
        if self.has_wrap:
            wrap = self.make_wrap(obj)
        else:
            wrap = None
        self.solidify_with_inside(obj, self.thickness)
        subsurf(obj, 2)
        handle_location = (
            x_anchors[-2] * (1 - self.handle_location)
            + x_anchors[-1] * self.handle_location,
            0,
            z_anchors[-2] * (1 - self.handle_location)
            + z_anchors[-1] * self.handle_location,
        )
        angle_low = np.arctan(
            (x_anchors[-1] - x_anchors[-2]) / (z_anchors[-1] - z_anchors[-2])
        )
        angle_height = np.arctan(
            (x_anchors[2] - x_anchors[1]) / (z_anchors[2] - z_anchors[1])
        )
        handle_angle = (
            self._handle_angle
            if self._use_fixed_spawn_draws
            else uniform(angle_low, angle_height + 1e-3)
        )
        if self.has_guard:
            obj = self.add_handle(obj, handle_location, handle_angle)
        if self.has_wrap:
            butil.select_none()
            obj = join_objects([obj, wrap])
        obj.scale = [self.scale] * 3
        butil.apply_transform(obj)
        return obj

    def add_handle(self, obj, handle_location, handle_angle):
        bpy.ops.mesh.primitive_torus_add(
            location=handle_location,
            major_radius=self.handle_radius,
            minor_radius=self.handle_inner_radius,
        )
        handle = bpy.context.active_object
        handle.rotation_euler = np.pi / 2, handle_angle, 0
        butil.modify_mesh(
            handle,
            "SIMPLE_DEFORM",
            deform_method="TAPER",
            angle=self.handle_taper_x,
            deform_axis="X",
        )
        butil.modify_mesh(
            handle,
            "SIMPLE_DEFORM",
            deform_method="TAPER",
            angle=self.handle_taper_y,
            deform_axis="Y",
        )
        butil.modify_mesh(handle, "BOOLEAN", object=obj, operation="DIFFERENCE")
        butil.select_none()
        objs = butil.split_object(handle)
        i = np.argmax([np.max(read_co(o)[:, 0]) for o in objs])
        handle = objs[i]
        objs.remove(handle)
        butil.delete(objs)
        subsurf(handle, 1)
        write_attribute(handle, lambda nw: 1, "guard", "FACE")
        return join_objects([obj, handle])

    def make_wrap(self, obj):
        butil.select_none()
        obj = deep_clone_obj(obj)
        remove_vertices(
            obj,
            lambda x, y, z: (z / self.depth < self.wrap_margin)
            | (
                z / self.depth
                > 1
                - self.wrap_margin
                + self._wrap_margin_jitter
            )
            | (np.abs(np.arctan2(y, x)) < np.pi * self.wrap_margin),
        )
        obj.scale = 1 + 1e-2, 1 + 1e-2, 1
        butil.apply_transform(obj)
        write_attribute(obj, lambda nw: 1, "text", "FACE")
        return obj

    def finalize_assets(self, assets):
        super().finalize_assets(assets)
        if self.has_wrap:
            for obj in assets if isinstance(assets, list) else [assets]:
                wrap_sides(obj, self.wrap_surface, "u", "v", "z", selection="text")
        if self.scratch:
            self.scratch.apply(assets)
        if self.edge_wear:
            self.edge_wear.apply(assets)
