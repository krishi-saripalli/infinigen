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
from infinigen.assets.materials import text
from infinigen.assets.utils.decorate import (
    geo_extension,
    read_co,
    subdivide_edge_ring,
    subsurf,
    write_co,
)
from infinigen.assets.utils.object import new_base_cylinder
from infinigen.assets.utils.uv import wrap_front_back
from infinigen.core import surface
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import AssetParameters, ParameterizedAssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform, weighted_sample


class FoodBagParameters(AssetParameters):
    is_packet: Annotated[
        bool, Field(json_schema_extra={"editable": True, "kind": "bool"})
    ]
    extrude_length: Annotated[
        float, Field(ge=0.05, le=0.1, json_schema_extra={"editable": True})
    ]


class FoodBagFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = FoodBagParameters

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_shape(self, length: float, is_packet: bool) -> tuple[float, float, float]:
        if is_packet:
            width = length * log_uniform(0.6, 1.0)
            depth = width * uniform(0.5, 0.8)
            curve_profile = uniform(2, 4)
        else:
            width = length * log_uniform(0.2, 0.4)
            depth = width * uniform(0.6, 1.0)
            curve_profile = uniform(4, 8)
        return width, depth, curve_profile

    def _sample_texture_shared(self, seed: int) -> bool:
        # NOTE: texture_shared is sampled on self in apply_parameters; excluded from quartet sampling (material-only, not exported geometry).
        with FixedSeed(seed):
            return bool(uniform() < 0.2)

    def _sample_init_parameters(self, seed: int) -> FoodBagParameters:
        is_packet = bool(uniform() < 0.6)
        return FoodBagParameters(
            seed=seed,
            is_packet=is_packet,
            extrude_length=uniform(0.05, 0.1),
        )

    def apply_parameters(
        self, params: FoodBagParameters, *, spawn_scope: bool = True
    ) -> None:
        is_packet = params.is_packet
        with FixedSeed(params.seed):
            self.length = uniform(0.1, 0.3)
            width, depth, curve_profile = self._sample_shape(self.length, is_packet)
            surface_mat = weighted_sample(material_assignments.graphicdesign)()()
            if surface_mat == text.Text:
                surface_mat = surface_mat(params.seed)
        self.is_packet = is_packet
        self.width = width
        self.depth = depth
        self.curve_profile = curve_profile
        self.extrude_length = params.extrude_length
        self.texture_shared = self._sample_texture_shared(params.seed)
        self.surface = surface_mat
        self._use_fixed_spawn_draws = spawn_scope

    def create_asset(self, **params) -> bpy.types.Object:
        obj = self.make_base()
        self.add_seal(obj)
        self.build_uv(obj)
        subsurf(obj, 2)
        surface.add_geomod(
            obj, geo_extension, input_kwargs={"musgrave_dimensions": "2D"}, apply=True
        )
        return obj

    def make_base(self):
        obj = new_base_cylinder()
        subdivide_edge_ring(obj, 64)
        obj.scale = self.width / 2, self.depth / 2, self.length / 2
        butil.apply_transform(obj)
        x, y, z = read_co(obj).T
        ratio = 1 - (2 * np.abs(z) / self.length) ** self.curve_profile
        write_co(obj, np.stack([x, ratio * y, z], -1))
        butil.modify_mesh(obj, "WELD", merge_threshold=1e-3)
        return obj

    def add_seal(self, obj):
        with butil.ViewportMode(obj, "EDIT"):
            bm = bmesh.from_edit_mesh(obj.data)
            for i in [-1, 1]:
                bpy.ops.mesh.select_all(action="DESELECT")
                bm.verts.ensure_lookup_table()
                indices = np.nonzero(read_co(obj)[:, -1] * i >= self.length / 2 - 1e-3)[
                    0
                ]
                for idx in indices:
                    bm.verts[idx].select_set(True)
                bm.select_flush(False)
                bmesh.update_edit_mesh(obj.data)
                bpy.ops.mesh.extrude_edges_move(
                    TRANSFORM_OT_translate={
                        "value": (0, 0, self.extrude_length * self.length * i)
                    }
                )

    def build_uv(self, obj):
        if not self.is_packet:
            obj.rotation_euler[1] = np.pi / 2
            butil.apply_transform(obj)
        wrap_front_back(obj, self.surface, self.texture_shared)
        if not self.is_packet:
            obj.rotation_euler[1] = -np.pi / 2
            butil.apply_transform(obj)
