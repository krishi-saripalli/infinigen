# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Annotated, ClassVar

import bmesh
import bpy
import numpy as np
from numpy.random import uniform
from pydantic import Field

from infinigen.assets.composition import material_assignments
from infinigen.assets.materials import text
from infinigen.assets.utils.decorate import read_co, subdivide_edge_ring, subsurf
from infinigen.assets.utils.draw import spin
from infinigen.assets.utils.object import join_objects, new_cylinder
from infinigen.assets.utils.uv import wrap_front_back
from infinigen.core import surface
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import (
    AssetParameters,
    ParameterizedAssetFactory,
)
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import weighted_sample


class BottleParameters(AssetParameters):
    x_length_ratio: Annotated[
        float, Field(ge=0.15, le=0.25, json_schema_extra={"editable": True})
    ]


class BottleFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = BottleParameters
    z_neck_offset = 0.05
    z_waist_offset = 0.15

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    # TODO: Sample bottle type and when branch-gated parameters are supported in sampling, sample those as well.
    def _sample_bottle_shape(self, seed: int, x_cap: float) -> None:
        with FixedSeed(seed):
            self.bottle_type = np.random.choice(
                ["beer", "bordeaux", "champagne", "coke", "vintage"]
            )
            match self.bottle_type:
                case "beer":
                    self.z_waist = 0.0
                    self.z_neck = uniform(0.5, 0.6)
                    self.z_cap = uniform(0.05, 0.08)
                    neck_size = uniform(0.06, 0.1)
                    neck_ratio = uniform(0.4, 0.5)
                    self.x_anchors = [
                        0,
                        1,
                        1,
                        (neck_ratio + 1) / 2 + (1 - neck_ratio) / 2 * x_cap,
                        neck_ratio + (1 - neck_ratio) * x_cap,
                        x_cap,
                        x_cap,
                        0,
                    ]
                    self.z_anchors = [
                        0,
                        0,
                        self.z_neck,
                        self.z_neck + uniform(0.6, 0.7) * neck_size,
                        self.z_neck + neck_size,
                        1 - self.z_cap,
                        1,
                        1,
                    ]
                    self.is_vector = [0, 1, 1, 0, 1, 1, 1, 0]
                case "bordeaux":
                    self.z_waist = 0.0
                    self.z_neck = uniform(0.6, 0.7)
                    self.z_cap = uniform(0.1, 0.15)
                    neck_size = uniform(0.1, 0.15)
                    self.x_anchors = (
                        0,
                        1,
                        1,
                        (1 + x_cap) / 2,
                        x_cap,
                        x_cap,
                        0,
                    )
                    self.z_anchors = [
                        0,
                        0,
                        self.z_neck,
                        self.z_neck + uniform(0.6, 0.7) * neck_size,
                        self.z_neck + neck_size,
                        1,
                        1,
                    ]
                    self.is_vector = [0, 1, 1, 0, 1, 1, 0]
                case "champagne":
                    self.z_waist = 0.0
                    self.z_neck = uniform(0.4, 0.5)
                    self.z_cap = uniform(0.05, 0.08)
                    self.x_anchors = [
                        0,
                        1,
                        1,
                        1,
                        (1 + x_cap) / 2,
                        x_cap,
                        x_cap,
                        0,
                    ]
                    self.z_anchors = [
                        0,
                        0,
                        self.z_neck,
                        self.z_neck + uniform(0.08, 0.1),
                        self.z_neck + uniform(0.15, 0.18),
                        1 - self.z_cap,
                        1,
                        1,
                    ]
                    self.is_vector = [0, 1, 1, 0, 0, 1, 1, 0]
                case "coke":
                    self.z_waist = uniform(0.4, 0.5)
                    self.z_neck = self.z_waist + uniform(0.2, 0.25)
                    self.z_cap = uniform(0.05, 0.08)
                    self.x_anchors = [
                        0,
                        uniform(0.85, 0.95),
                        1,
                        uniform(0.85, 0.95),
                        1,
                        1,
                        x_cap,
                        x_cap,
                        0,
                    ]
                    self.z_anchors = [
                        0,
                        0,
                        uniform(0.08, 0.12),
                        uniform(0.18, 0.25),
                        self.z_waist,
                        self.z_neck,
                        1 - self.z_cap,
                        1,
                        1,
                    ]
                    self.is_vector = [0, 1, 0, 0, 1, 1, 1, 1, 0]
                case "vintage":
                    self.z_waist = uniform(0.1, 0.15)
                    self.z_neck = uniform(0.7, 0.75)
                    self.z_cap = uniform(0.0, 0.08)
                    x_lower = uniform(0.85, 0.95)
                    self.x_anchors = [
                        0,
                        x_lower,
                        (x_lower + 1) / 2,
                        1,
                        1,
                        (x_cap + 1) / 2,
                        x_cap,
                        x_cap,
                        0,
                    ]
                    self.z_anchors = [
                        0,
                        0,
                        self.z_waist - uniform(0.1, 0.15),
                        self.z_waist,
                        self.z_neck,
                        self.z_neck + uniform(0.1, 0.2),
                        1 - self.z_cap,
                        1,
                        1,
                    ]
                    self.is_vector = [0, 1, 0, 1, 1, 0, 1, 1, 0]
            self.x_cap = x_cap

    def _sample_materials(self, seed: int) -> None:
        with FixedSeed(seed):
            self.surface = weighted_sample(material_assignments.plastics)()()
            wrap_surface = text.Text()()
            if wrap_surface == text.Text:
                wrap_surface = text.Text(False)
            self.wrap_surface = wrap_surface
            self.cap_surface = weighted_sample(material_assignments.metals)()()

    def _sample_texture_shared(self, seed: int) -> bool:
        # NOTE: texture_shared is sampled on self in apply_parameters; excluded from quartet sampling (material-only, not exported geometry).
        with FixedSeed(seed):
            return bool(uniform() < 0.2)

    def _sample_init_parameters(self, seed: int) -> BottleParameters:
        x_cap = uniform(0.3, 0.35)
        self._sample_bottle_shape(seed, x_cap)
        self._sample_materials(seed)
        self.texture_shared = self._sample_texture_shared(seed)
        return BottleParameters(
            seed=seed,
            x_length_ratio=uniform(0.15, 0.25),
        )

    def apply_parameters(
        self, params: BottleParameters, *, spawn_scope: bool = True
    ) -> None:
        # NOTE: x_cap does not elicit a clear visual change in exported geometry; excluded from quartet sampling.
        with FixedSeed(params.seed):
            self.x_cap = uniform(0.3, 0.35)
        self._sample_bottle_shape(params.seed, self.x_cap)
        self._sample_materials(params.seed)
        # NOTE: z_length sampled on self from seed; excluded from quartet sampling (uniform scale normalized away in point clouds).
        with FixedSeed(params.seed):
            self.z_length = uniform(0.15, 0.25)
        self.x_length = self.z_length * params.x_length_ratio
        # NOTE: bottle_width and cap_subsurf do not elicit a clear visual change in exported geometry; excluded from quartet sampling.
        with FixedSeed(params.seed):
            self.bottle_width = uniform(0.002, 0.005)
            self.cap_subsurf = bool(uniform() < 0.5)
        self.texture_shared = self._sample_texture_shared(params.seed)
        self._use_fixed_spawn_draws = spawn_scope
        if spawn_scope:
            with FixedSeed(params.seed):
                wrap_z_max_frac = uniform(0.02, self.z_neck_offset)
                wrap_z_min_frac = uniform(0.02, self.z_waist_offset)
            self.wrap_z_max = self.z_neck - wrap_z_max_frac * (
                self.z_neck - self.z_waist
            )
            self.wrap_z_min = self.z_waist + wrap_z_min_frac * (
                self.z_neck - self.z_waist
            )
        else:
            self.wrap_z_max = None
            self.wrap_z_min = None

    def create_asset(self, **params) -> bpy.types.Object:
        bottle = self.make_bottle()
        wrap = self.make_wrap(bottle)
        cap = self.make_cap()
        obj = join_objects([bottle, wrap, cap])

        return obj

    def finalize_assets(self, assets):
        pass

    def make_bottle(self):
        x_anchors = np.array(self.x_anchors) * self.x_length
        z_anchors = np.array(self.z_anchors) * self.z_length
        anchors = x_anchors, 0, z_anchors
        obj = spin(anchors, np.nonzero(self.is_vector)[0])
        subsurf(obj, 1)
        if self.bottle_width > 0:
            butil.modify_mesh(obj, "SOLIDIFY", thickness=self.bottle_width)

        surface.assign_material(obj, self.surface)

        return obj

    def make_wrap(self, bottle):
        obj = new_cylinder(vertices=128)
        with butil.ViewportMode(obj, "EDIT"):
            bm = bmesh.from_edit_mesh(obj.data)
            geom = [f for f in bm.faces if len(f.verts) > 4]
            bmesh.ops.delete(bm, geom=geom, context="FACES_ONLY")
            bmesh.update_edit_mesh(obj.data)
        subdivide_edge_ring(obj, 16)
        z_max = (
            self.wrap_z_max
            if self._use_fixed_spawn_draws
            else self.z_neck
            - uniform(0.02, self.z_neck_offset) * (self.z_neck - self.z_waist)
        )
        z_min = (
            self.wrap_z_min
            if self._use_fixed_spawn_draws
            else self.z_waist
            + uniform(0.02, self.z_waist_offset) * (self.z_neck - self.z_waist)
        )
        radius = np.max(read_co(bottle)[:, 0]) + 2e-3
        obj.scale = radius, radius, (z_max - z_min) * self.z_length
        obj.location[-1] = z_min * self.z_length
        butil.apply_transform(obj, True)
        wrap_front_back(obj, self.wrap_surface, self.texture_shared)
        return obj

    def make_cap(self):
        obj = new_cylinder(vertices=128)
        obj.scale = [
            (self.x_cap + 0.1) * self.x_length,
            (self.x_cap + 0.1) * self.x_length,
            (self.z_cap + 0.01) * self.z_length,
        ]
        obj.location[-1] = (1 - self.z_cap) * self.z_length
        butil.apply_transform(obj, loc=True)
        subsurf(obj, 1, self.cap_subsurf)
        surface.assign_material(obj, self.cap_surface)
        return obj
