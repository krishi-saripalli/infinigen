# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from __future__ import annotations

from typing import Annotated, ClassVar

import bpy
from numpy.random import uniform
from pydantic import Field

from infinigen.assets.composition import material_assignments
from infinigen.assets.materials.art import ArtRug
from infinigen.assets.utils.object import new_base_circle, new_bbox, new_plane
from infinigen.assets.utils.uv import wrap_sides
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import (
    AssetParameters,
    ParameterizedAssetFactory,
)
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import clip_gaussian, weighted_sample


class RugParameters(AssetParameters):
    rug_shape: Annotated[
        str,
        Field(
            json_schema_extra={
                "editable": False,
                "kind": "enum",
                "choices": ["rectangle", "circle", "rounded", "ellipse"],
            }
        ),
    ] = "rounded"
    length: Annotated[float, Field(ge=1.0, le=1.5, json_schema_extra={"editable": False})]


class RugFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = RugParameters

    def __init__(self, factory_seed, coarse=False):
        super(RugFactory, self).__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _apply_internal_state(self, params: RugParameters) -> None:
        with FixedSeed(params.seed):
            clip_gaussian(3, 1, 2, 6)
            if self.rug_shape != "circle":
                uniform(1, 1.5)
            uniform(0.1, 0.5)
            uniform(0.01, 0.02)
            surface_gen_class = weighted_sample(material_assignments.rug_fabric)
            surface = surface_gen_class()()
            if surface == ArtRug:
                surface = surface(params.seed)
            self.surface = surface

    def _sample_init_parameters(self, seed: int) -> RugParameters:
        with FixedSeed(seed):
            rug_shape = "rounded"
            length = uniform(1, 1.5)
            surface_gen_class = weighted_sample(material_assignments.rug_fabric)
            surface = surface_gen_class()()
            if surface == ArtRug:
                surface = surface(seed)
            self.surface = surface
        return RugParameters(
            seed=seed,
            rug_shape=rug_shape,
            length=length,
        )

    def apply_parameters(
        self, params: RugParameters, *, spawn_scope: bool = True
    ) -> None:
        self.rug_shape = params.rug_shape
        # NOTE: width sampled on self from seed; excluded from quartet sampling (uniform scale normalized away in point clouds).
        with FixedSeed(params.seed):
            self.width = clip_gaussian(3, 1, 2, 6)
        self._apply_internal_state(params)
        # NOTE: length only scales non-circle rugs; circle shape uses width for both axes.
        self.length = (
            self.width
            if self.rug_shape == "circle"
            else self.width * params.length
        )
        # NOTE: rounded_buffer and thickness sampled on self from seed; excluded from quartet sampling.
        with FixedSeed(params.seed):
            self.rounded_buffer = uniform(0.1, 0.5)
            self.thickness = uniform(0.01, 0.02)
        self._use_fixed_spawn_draws = spawn_scope

    def build_shape(self):
        match self.rug_shape:
            case "rectangle":
                obj = new_plane()
                obj.scale = self.length / 2, self.width / 2, 1
                butil.apply_transform(obj, True)
            case "rounded":
                obj = new_plane()
                obj.scale = self.length / 2, self.width / 2, 1
                butil.apply_transform(obj, True)
                butil.modify_mesh(
                    obj, "BEVEL", width=self.rounded_buffer * self.width, segments=16
                )
            case _:
                obj = new_base_circle(vertices=128)
                with butil.ViewportMode(obj, "EDIT"):
                    bpy.ops.mesh.select_all(action="SELECT")
                    bpy.ops.mesh.edge_face_add()
                obj.scale = self.length / 2, self.width / 2, 1
                butil.apply_transform(obj, True)
        return obj

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        return new_bbox(
            -self.length / 2,
            self.length / 2,
            -self.width / 2,
            self.width / 2,
            0,
            self.thickness,
        )

    def create_asset(self, **params) -> bpy.types.Object:
        obj = self.build_shape()
        wrap_sides(obj, self.surface, "z", "x", "y")
        butil.modify_mesh(obj, "SOLIDIFY", thickness=self.thickness, offset=1)
        return obj
