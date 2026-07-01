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

from infinigen.assets.objects.tableware.base import (
    TablewareFactory,
    apply_tableware_from_draws,
    sample_tableware_base,
)
from infinigen.assets.utils.decorate import subsurf, write_co
from infinigen.assets.utils.object import new_grid
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import (
    AssetParameters,
    ParameterizedAssetFactory,
)
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform


class KnifeParameters(AssetParameters):
    x_length: Annotated[float, Field(ge=0.4, le=0.7, json_schema_extra={"editable": False})]
    y_length: Annotated[float, Field(ge=0.1, le=0.5, json_schema_extra={"editable": True})]
    thickness: Annotated[float, Field(ge=0.02, le=0.03, json_schema_extra={"editable": False})]
    has_guard: Annotated[
        bool, Field(json_schema_extra={"editable": False, "kind": "bool"})
    ] = True
    # NOTE: round vs double guard geometry branch.
    guard_type: Annotated[
        str,
        Field(
            json_schema_extra={
                "editable": False,
                "kind": "enum",
                "choices": ["round", "double"],
            }
        ),
    ] = "round"
    x_anchor_1: Annotated[
        float, Field(ge=0.5, le=0.8, json_schema_extra={"editable": True})
    ] = 0.65
    x_anchor_2: Annotated[
        float, Field(ge=0.3, le=0.4, json_schema_extra={"editable": True})
    ] = 0.35
    y_anchor_1_mult: Annotated[
        float, Field(ge=0.75, le=0.95, json_schema_extra={"editable": True})
    ] = 0.85

class KnifeFactory(ParameterizedAssetFactory, TablewareFactory):
    parameters_model: ClassVar[type[AssetParameters]] = KnifeParameters
    x_end = 0.5

    def __init__(self, factory_seed, coarse=False):
        AssetFactory.__init__(self, factory_seed, coarse)
        self.init_legacy_parameters()

    def _apply_knife_branches(self) -> None:
        y_off = self.y_off_rand
        if y_off < 1 / 8:
            self.y_offset = 0.2
        elif y_off < 1 / 4:
            self.y_offset = 0.5
        else:
            self.y_offset = 0.2 + (y_off - 0.25) / 0.75 * 0.4

    def _sample_init_parameters(self, seed: int) -> KnifeParameters:
        thickness = log_uniform(0.02, 0.03)
        has_guard = True
        y_length = log_uniform(0.1, 0.5)
        return KnifeParameters(
            seed=seed,
            x_length=log_uniform(0.4, 0.7),
            y_length=y_length,
            thickness=thickness,
            has_guard=has_guard,
            guard_type="round" if uniform(0, 1) < 0.6 else "double",
            x_anchor_1=uniform(0.5, 0.8),
            x_anchor_2=uniform(0.3, 0.4),
            y_anchor_1_mult=log_uniform(0.75, 0.95),
        )

    def _sample_spawn_parameters(
        self, params: KnifeParameters, seed: int, i: int
    ) -> KnifeParameters:
        return params.model_copy(
            update={
                "x_anchor_1": uniform(0.5, 0.8),
                "x_anchor_2": uniform(0.3, 0.4),
                "y_anchor_1_mult": log_uniform(0.75, 0.95),
            }
        )

    def apply_parameters(
        self, params: KnifeParameters, *, spawn_scope: bool = True
    ) -> None:
        # NOTE: scale sampled on self from seed; excluded from quartet sampling (uniform scale normalized away in point clouds).
        with FixedSeed(params.seed):
            self.scale = log_uniform(0.2, 0.3)
            base = sample_tableware_base(params.seed)
            self._lower_thresh = base["lower_thresh"]
            guard_depth_mult = log_uniform(0.2, 1.0)
            y_guard_ratio = log_uniform(0.2, 0.4)
            self.y_off_rand = uniform(0, 1)
        apply_tableware_from_draws(
            self,
            seed=params.seed,
            lower_thresh=self._lower_thresh,
            scale=self.scale,
            guard_depth=guard_depth_mult * params.thickness,
        )
        self.has_guard = params.has_guard
        self.guard_type = params.guard_type
        self.y_length = params.y_length
        self.y_guard = params.y_length * y_guard_ratio
        self._apply_knife_branches()
        self.x_length = params.x_length
        # NOTE: x_guard and has_tip do not elicit a clear visual change in exported geometry; excluded from quartet sampling.
        with FixedSeed(params.seed):
            self.x_guard = uniform(0, 0.2)
            self.has_tip = bool(uniform(0, 1) < 0.7)
        self.thickness = params.thickness
        self.guard_depth = guard_depth_mult * params.thickness
        self._x_anchor_1 = params.x_anchor_1
        self._x_anchor_2 = params.x_anchor_2
        self._y_anchor_1_mult = params.y_anchor_1_mult
        self._use_fixed_spawn_draws = spawn_scope

    def create_asset(self, **params) -> bpy.types.Object:
        if self._use_fixed_spawn_draws:
            x1 = self._x_anchor_1
            x2 = self._x_anchor_2
            y1 = self.y_length * self._y_anchor_1_mult
        else:
            x1 = uniform(0.5, 0.8)
            x2 = uniform(0.3, 0.4)
            y1 = self.y_length * log_uniform(0.75, 0.95)
        x_anchors = np.array(
            [
                self.x_end,
                x1 * self.x_end,
                x2 * self.x_end,
                1e-3,
                0,
                -1e-3,
                -2e-3,
                -self.x_end * self.x_length + 1e-3,
                -self.x_end * self.x_length,
            ]
        )
        y_anchors = np.array(
            [
                1e-3,
                y1,
                self.y_length,
                self.y_length,
                self.y_length,
                self.y_guard,
                self.y_guard,
                self.y_guard,
                self.y_guard,
            ]
        )
        if not self.has_guard:
            indices = [0, 1, 2, 4, 5, 7, 8]
            x_anchors = x_anchors[indices]
            y_anchors = y_anchors[indices]
        if self.has_tip:
            indices = [0] + list(range(len(x_anchors)))
            x_anchors = x_anchors[indices]
            x_anchors[0] += 1e-3
            y_anchors = y_anchors[indices]
            y_anchors[1] += 3e-3

        obj = new_grid(x_subdivisions=len(x_anchors) - 1, y_subdivisions=1)
        x = np.concatenate([x_anchors] * 2)
        y = np.concatenate([y_anchors, np.zeros_like(y_anchors)])
        y[0 :: len(y_anchors)] += self.y_offset * self.y_length
        if self.has_tip:
            y[1 :: len(y_anchors)] += self.y_offset * self.y_length
            y[2 :: len(y_anchors)] += self.y_offset * (self.y_length - y_anchors[2])
        else:
            y[1 :: len(y_anchors)] += self.y_offset * (self.y_length - y_anchors[1])
        z = np.concatenate([np.zeros_like(x_anchors)] * 2)
        write_co(obj, np.stack([x, y, z], -1))
        butil.modify_mesh(obj, "SOLIDIFY", thickness=self.thickness)
        self.make_knife_tip(obj)
        subsurf(obj, 1)

        def selection(nw, x):
            return nw.compare(
                "LESS_THAN", x, -self.x_guard * self.x_length * self.x_end
            )

        if self.guard_type == "double":
            selection = self.make_double_sided(selection)
        self.add_guard(obj, selection)
        subsurf(obj, 1)
        subsurf(obj, 1, True)
        obj.scale = [self.scale] * 3
        butil.apply_transform(obj)
        return obj

    def make_knife_tip(self, obj):
        with butil.ViewportMode(obj, "EDIT"):
            bm = bmesh.from_edit_mesh(obj.data)
            for e in bm.edges:
                u, v = e.verts
                x0, y0, z0 = u.co
                x1, y1, z1 = v.co
                if x0 >= 0 and x1 >= 0 and abs(x0 - x1) < 2e-4:
                    if (
                        y0 > self.y_offset * self.y_length
                        and y1 > self.y_offset * self.y_length
                    ):
                        bmesh.ops.pointmerge(
                            bm, verts=[u, v], merge_co=(u.co + v.co) / 2
                        )
            bmesh.update_edit_mesh(obj.data)
            bpy.ops.mesh.select_mode(type="EDGE")
            bpy.ops.mesh.select_loose(extend=False)
            bpy.ops.mesh.delete(type="EDGE")
