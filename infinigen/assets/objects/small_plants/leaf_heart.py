# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Beining Han

from __future__ import annotations

from typing import Annotated, ClassVar

import bpy
import numpy as np
from pydantic import Field

from infinigen.assets.objects.trees.utils import mesh
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import (
    AssetParameters,
    ParameterizedAssetFactory,
)
from infinigen.core.tagging import tag_object
from infinigen.core.util import blender as butil


class LeafHeartParameters(AssetParameters):
    use_wave: Annotated[
        bool, Field(json_schema_extra={"editable": False, "kind": "bool"})
    ] = True
    leaf_width: Annotated[
        float, Field(ge=0.7, le=1.3, json_schema_extra={"editable": True})
    ] = 1.0
    z_scaling: Annotated[
        float, Field(ge=-0.1, le=0.1, json_schema_extra={"editable": True})
    ] = 0.0
    width_noise: Annotated[
        float, Field(ge=-1.0, le=1.0, json_schema_extra={"editable": False})
    ] = 0.0
    wave_height: Annotated[
        float, Field(ge=-0.64, le=0.64, json_schema_extra={"editable": False})
    ] = 0.0
    wave_width: Annotated[
        float, Field(ge=2.5, le=4.5, json_schema_extra={"editable": False})
    ] = 3.5
    wave_speed: Annotated[
        float, Field(ge=30.0, le=60.0, json_schema_extra={"editable": False})
    ] = 40.0


class LeafHeartFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = LeafHeartParameters
    scale = 0.2

    def __init__(self, factory_seed, genome: dict | None = None, coarse=False):
        super(LeafHeartFactory, self).__init__(factory_seed, coarse=coarse)
        self._genome_override = genome
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> LeafHeartParameters:
        genome = self._genome_override or {}
        params = LeafHeartParameters(
            seed=seed, use_wave=True
        )
        if self._genome_override:
            updates = {
                k: v
                for k, v in self._genome_override.items()
                if k in LeafHeartParameters.model_fields
            }
            params = params.model_copy(update=updates)
        return params

    def _sample_spawn_parameters(
        self, params: LeafHeartParameters, seed: int, i: int
    ) -> LeafHeartParameters:
        return params.model_copy(
            update={
                "width_noise": float(np.random.randn()),
                "wave_height": float(0.8 * np.random.randn() * 0.8),
                "wave_width": float(3.5 + np.random.randn() * 1.0),
                "wave_speed": float(40 + np.random.uniform(-10, 20)),
            }
        )

    def apply_parameters(
        self, params: LeafHeartParameters, *, spawn_scope: bool = True
    ) -> None:
        genome_override = self._genome_override or {}
        self.genome = {
            "leaf_width": params.leaf_width,
            "use_wave": params.use_wave,
            "z_scaling": params.z_scaling,
            "width_rand": genome_override.get("width_rand", 0.1),
        }
        self.width_noise = params.width_noise
        self.wave_height = params.wave_height
        self.wave_width = params.wave_width
        self.wave_speed = params.wave_speed
        self._use_fixed_spawn_draws = spawn_scope

    def create_asset(self, **params) -> bpy.types.Object:
        bpy.ops.mesh.primitive_circle_add(
            enter_editmode=False, align="WORLD", location=(0, 0, 0), scale=(1, 1, 1)
        )
        bpy.ops.object.editmode_toggle()
        bpy.ops.mesh.edge_face_add()

        obj = bpy.context.active_object
        n = len(obj.data.vertices) // 2

        mesh.select_vtx_by_idx(obj, [0, -1], deselect=True)
        bpy.ops.mesh.subdivide()

        a = np.linspace(0, np.pi, n)
        width_noise = (
            self.width_noise if self._use_fixed_spawn_draws else float(np.random.randn())
        )
        x = (
            16.0
            * (np.sin(a - np.pi) ** 3)
            * (self.genome["leaf_width"] + width_noise * self.genome["width_rand"])
        )
        y = (
            13.0 * np.cos(a - np.pi)
            - 5 * np.cos(2 * (a - np.pi))
            - 2 * np.cos(3 * (a - np.pi))
        )
        x, y = x * 0.3, y * 0.3
        z = x**2 * self.genome["z_scaling"]

        full_coords = np.concatenate(
            [
                np.stack([x, y, z], 1),
                np.stack([-x[::-1], y[::-1], z], 1),
                np.array([[0, y[0], 0]]),
            ]
        ).flatten()
        bpy.ops.object.mode_set(mode="OBJECT")
        obj.data.vertices.foreach_set("co", full_coords)

        if self.genome["use_wave"]:
            wave_height = (
                self.wave_height
                if self._use_fixed_spawn_draws
                else float(0.8 * np.random.randn() * 0.8)
            )
            wave_width = (
                self.wave_width
                if self._use_fixed_spawn_draws
                else float(3.5 + np.random.randn() * 1.0)
            )
            wave_speed = (
                self.wave_speed
                if self._use_fixed_spawn_draws
                else float(40 + np.random.uniform(-10, 20))
            )
            bpy.ops.object.modifier_add(type="WAVE")
            bpy.context.object.modifiers["Wave"].height = wave_height
            bpy.context.object.modifiers["Wave"].width = wave_width
            bpy.context.object.modifiers["Wave"].speed = wave_speed

        mesh.finalize_obj(obj)
        butil.modify_mesh(
            obj,
            "SOLIDIFY",
            thickness=0.01,
            offset=0,
            use_even_offset=True,
        )
        bpy.context.scene.cursor.location = obj.data.vertices[-1].co

        bpy.ops.object.origin_set(type="ORIGIN_CURSOR")

        obj.location = (0, 0, 0)
        obj.scale *= self.scale
        butil.apply_transform(obj)
        tag_object(obj, "leaf_heart")

        return obj
