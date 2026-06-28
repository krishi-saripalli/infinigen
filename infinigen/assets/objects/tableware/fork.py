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


class ForkParameters(AssetParameters):
    # NOTE: tine layout branches on standard_tines/n_cuts sampled at init.
    x_tip: Annotated[float, Field(ge=0.15, le=0.2, json_schema_extra={"editable": False})]
    y_length: Annotated[float, Field(ge=0.05, le=0.08, json_schema_extra={"editable": False})]
    z_depth: Annotated[float, Field(ge=0.02, le=0.04, json_schema_extra={"editable": False})]
    thickness: Annotated[float, Field(ge=0.008, le=0.015, json_schema_extra={"editable": False})]
    # NOTE: guard depth only applied when guard_type=double (sampled branch).
    guard_depth_mult: Annotated[
        float, Field(ge=0.2, le=1.0, json_schema_extra={"editable": False})
    ]
    has_guard: Annotated[
        bool, Field(json_schema_extra={"editable": False, "kind": "bool"})
    ] = True
    lower_thresh: Annotated[float, Field(ge=0.5, le=0.8, json_schema_extra={"editable": False})]
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
    standard_tines: Annotated[
        bool, Field(json_schema_extra={"editable": True, "kind": "bool"})
    ] = True
    x_anchor_1: Annotated[
        float, Field(ge=-0.04, le=-0.02, json_schema_extra={"editable": True})
    ] = -0.03
    x_anchor_tail_mult: Annotated[
        float, Field(ge=1.2, le=1.4, json_schema_extra={"editable": True})
    ] = 1.3
    y_anchor_0_mult: Annotated[
        float, Field(ge=0.8, le=1.0, json_schema_extra={"editable": True})
    ] = 0.9
    y_anchor_1_mult: Annotated[
        float, Field(ge=1.0, le=1.2, json_schema_extra={"editable": True})
    ] = 1.1
    y_anchor_2_mult: Annotated[
        float, Field(ge=0.6, le=1.0, json_schema_extra={"editable": True})
    ] = 0.8
    y_anchor_3_mult: Annotated[
        float, Field(ge=0.2, le=0.4, json_schema_extra={"editable": True})
    ] = 0.3
    y_anchor_4: Annotated[
        float, Field(ge=0.01, le=0.02, json_schema_extra={"editable": True})
    ] = 0.015
    y_anchor_5: Annotated[
        float, Field(ge=0.02, le=0.05, json_schema_extra={"editable": True})
    ] = 0.035
    y_anchor_6: Annotated[
        float, Field(ge=0.01, le=0.02, json_schema_extra={"editable": True})
    ] = 0.015
    z_anchor_mid: Annotated[
        float, Field(ge=-0.02, le=0.04, json_schema_extra={"editable": True})
    ] = 0.01
    z_anchor_tail: Annotated[
        float, Field(ge=-0.02, le=0.0, json_schema_extra={"editable": True})
    ] = -0.01


class ForkFactory(ParameterizedAssetFactory, TablewareFactory):
    parameters_model: ClassVar[type[AssetParameters]] = ForkParameters
    x_end = 0.15
    is_fragile = True

    def _sample_spawn_anchor_updates(self) -> dict[str, float]:
        return {
            "x_anchor_1": uniform(-0.04, -0.02),
            "x_anchor_tail_mult": log_uniform(1.2, 1.4),
            "y_anchor_0_mult": log_uniform(0.8, 1.0),
            "y_anchor_1_mult": log_uniform(1.0, 1.2),
            "y_anchor_2_mult": log_uniform(0.6, 1.0),
            "y_anchor_3_mult": log_uniform(0.2, 0.4),
            "y_anchor_4": log_uniform(0.01, 0.02),
            "y_anchor_5": log_uniform(0.02, 0.05),
            "y_anchor_6": log_uniform(0.01, 0.02),
            "z_anchor_mid": uniform(-0.02, 0.04),
            "z_anchor_tail": uniform(-0.02, 0.0),
        }

    def __init__(self, factory_seed, coarse=False):
        AssetFactory.__init__(self, factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_branch_state(self, seed: int) -> None:
        with FixedSeed(seed):
            self.has_cut = True

    def _sample_init_parameters(self, seed: int) -> ForkParameters:
        base = sample_tableware_base(seed)
        thickness = log_uniform(0.008, 0.015)
        self._sample_branch_state(seed)
        standard_tines = bool(uniform() >= 0.3)
        return ForkParameters(
            seed=seed,
            x_tip=uniform(0.15, 0.2),
            y_length=log_uniform(0.05, 0.08),
            z_depth=log_uniform(0.02, 0.04),
            thickness=thickness,
            guard_depth_mult=log_uniform(0.2, 1.0),
            has_guard=True,
            lower_thresh=base["lower_thresh"],
            scratch_draw=base["scratch_draw"],
            edge_wear_draw=base["edge_wear_draw"],
            standard_tines=standard_tines,
            **self._sample_spawn_anchor_updates(),
        )

    def _sample_spawn_parameters(
        self, params: ForkParameters, seed: int, i: int
    ) -> ForkParameters:
        return params.model_copy(update=self._sample_spawn_anchor_updates())

    def apply_parameters(
        self, params: ForkParameters, *, spawn_scope: bool = True
    ) -> None:
        # NOTE: scale sampled on self from seed; excluded from quartet sampling (uniform scale normalized away in point clouds).
        with FixedSeed(params.seed):
            self.scale = log_uniform(0.15, 0.25)
            # NOTE: x_length, z_offset do not elicit a clear visual change in exported geometry; excluded from quartet sampling.
            self.x_length = log_uniform(0.4, 0.8)
            self.z_offset = uniform(0.0, 0.05)
        apply_tableware_from_draws(
            self,
            seed=params.seed,
            lower_thresh=params.lower_thresh,
            scale=self.scale,
            scratch_draw=params.scratch_draw,
            edge_wear_draw=params.edge_wear_draw,
            guard_depth=params.guard_depth_mult * params.thickness,
        )
        self._sample_branch_state(params.seed)
        with FixedSeed(params.seed):
            self.n_cuts = (
                3 if params.standard_tines else int(np.random.randint(1, 3))
            )
        self.x_tip = params.x_tip
        self.y_length = params.y_length
        self.z_depth = params.z_depth
        self.thickness = params.thickness
        self.has_guard = params.has_guard
        # NOTE: guard_type does not elicit a clear visual change in exported geometry; excluded from quartet sampling.
        with FixedSeed(params.seed):
            self.guard_type = "round" if uniform(0, 1) < 0.6 else "double"
        self.guard_depth = params.guard_depth_mult * params.thickness
        self._x_anchor_1 = params.x_anchor_1
        self._x_anchor_tail_mult = params.x_anchor_tail_mult
        self._y_anchor_0_mult = params.y_anchor_0_mult
        self._y_anchor_1_mult = params.y_anchor_1_mult
        self._y_anchor_2_mult = params.y_anchor_2_mult
        self._y_anchor_3_mult = params.y_anchor_3_mult
        self._y_anchor_4 = params.y_anchor_4
        self._y_anchor_5 = params.y_anchor_5
        self._y_anchor_6 = params.y_anchor_6
        self._z_anchor_mid = params.z_anchor_mid
        self._z_anchor_tail = params.z_anchor_tail
        self._use_fixed_spawn_draws = spawn_scope

    def create_asset(self, **params) -> bpy.types.Object:
        if self._use_fixed_spawn_draws:
            x1 = self._x_anchor_1
            x_tail_mult = self._x_anchor_tail_mult
            y0 = self.y_length * self._y_anchor_0_mult
            y1 = self.y_length * self._y_anchor_1_mult
            y2 = self.y_length * self._y_anchor_2_mult
            y3 = self.y_length * self._y_anchor_3_mult
            y4 = self._y_anchor_4
            y5 = self._y_anchor_5
            y6 = self._y_anchor_6
            z_mid = self._z_anchor_mid
            z_tail = self._z_anchor_tail
        else:
            x1 = uniform(-0.04, -0.02)
            x_tail_mult = log_uniform(1.2, 1.4)
            y0 = self.y_length * log_uniform(0.8, 1.0)
            y1 = self.y_length * log_uniform(1.0, 1.2)
            y2 = self.y_length * log_uniform(0.6, 1.0)
            y3 = self.y_length * log_uniform(0.2, 0.4)
            y4 = log_uniform(0.01, 0.02)
            y5 = log_uniform(0.02, 0.05)
            y6 = log_uniform(0.01, 0.02)
            z_mid = uniform(-0.02, 0.04)
            z_tail = uniform(-0.02, 0.0)
        x_anchors = np.array(
            [
                self.x_tip,
                x1,
                -0.08,
                -0.12,
                -self.x_end,
                -self.x_end - self.x_length,
                -self.x_end - self.x_length * x_tail_mult,
            ]
        )
        y_anchors = np.array([y0, y1, y2, y3, y4, y5, y6])
        z_anchors = np.array(
            [
                0,
                -self.z_depth,
                -self.z_depth,
                0,
                self.z_offset,
                self.z_offset + z_mid,
                self.z_offset + z_tail,
            ]
        )
        n = 2 * (self.n_cuts + 1)
        obj = new_grid(x_subdivisions=len(x_anchors) - 1, y_subdivisions=n - 1)
        x = np.concatenate([x_anchors] * n)
        y = np.ravel(y_anchors[np.newaxis, :] * np.linspace(1, -1, n)[:, np.newaxis])
        z = np.concatenate([z_anchors] * n)
        write_co(obj, np.stack([x, y, z], -1))
        if self.has_cut:
            self.make_cuts(obj)
        butil.modify_mesh(obj, "SOLIDIFY", thickness=self.thickness)
        subsurf(obj, 1)

        def selection(nw, x):
            return nw.compare("LESS_THAN", x, -self.x_end)

        if self.guard_type == "double":
            selection = self.make_double_sided(selection)
        self.add_guard(obj, selection)
        subsurf(obj, 1)
        obj.scale = [self.scale] * 3
        butil.apply_transform(obj)
        return obj

    def make_cuts(self, obj):
        with butil.ViewportMode(obj, "EDIT"):
            bm = bmesh.from_edit_mesh(obj.data)
            front_verts = []
            for v in bm.verts:
                if abs(v.co[0] - self.x_tip) < 1e-3:
                    front_verts.append(v)
            front_verts = sorted(front_verts, key=lambda v: v.co[1])
            geom = []
            for f in bm.faces:
                vs = list(v for v in f.verts if v in front_verts)
                if len(vs) == 2:
                    if min(front_verts.index(vs[0]), front_verts.index(vs[1])) % 2 == 1:
                        geom.append(f)
            bmesh.ops.delete(bm, geom=geom, context="FACES")
            bmesh.update_edit_mesh(obj.data)


class SpatulaFactory(ForkFactory):
    def __init__(self, factory_seed, coarse=False):
        super(SpatulaFactory, self).__init__(factory_seed, coarse)
        self.has_cut = False
        self.z_depth = uniform(0, 0.05)
        self.y_length = log_uniform(0.08, 0.12)
