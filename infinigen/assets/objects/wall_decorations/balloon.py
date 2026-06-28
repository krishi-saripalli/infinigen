# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from __future__ import annotations

from typing import Annotated, Any, ClassVar

import bpy
import numpy as np
from numpy.random import uniform
from pydantic import Field

from infinigen.assets.composition import material_assignments
from infinigen.assets.scatters import clothes
from infinigen.assets.utils.decorate import subdivide_edge_ring, subsurf
from infinigen.assets.utils.draw import remesh_fill
from infinigen.assets.utils.misc import generate_text
from infinigen.assets.utils.object import new_bbox
from infinigen.core import surface
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import AssetParameters, ParameterizedAssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import weighted_sample


class BalloonParameters(AssetParameters):
    thickness: Annotated[float, Field(ge=0.06, le=0.1, json_schema_extra={"editable": False})]
    tension_stiffness: Annotated[
        float, Field(ge=0.0, le=5.0, json_schema_extra={"editable": True})
    ] = 0.0
    uniform_pressure_force: Annotated[
        float, Field(ge=10.0, le=20.0, json_schema_extra={"editable": True})
    ] = 15.0


class BalloonFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = BalloonParameters
    alpha = 0.8

    def __init__(
        self,
        factory_seed,
        material_gen=None,
        coarse=False,
    ):
        super(BalloonFactory, self).__init__(factory_seed, coarse)
        self._material_gen_override = material_gen
        self.init_legacy_parameters()

    def _sample_materials(self) -> tuple[Any, Any]:
        material_gen = self._material_gen_override
        if material_gen is None:
            material_gen = weighted_sample(material_assignments.decorative_metal)()
        return material_gen, material_gen()

    def _sample_spawn_field_updates(self) -> dict[str, float]:
        return {
            "tension_stiffness": uniform(0, 5),
            "uniform_pressure_force": uniform(10, 20),
        }

    def _sample_init_parameters(self, seed: int) -> BalloonParameters:
        self._material_gen, self._surface = self._sample_materials()
        return BalloonParameters(
            seed=seed,
            thickness=uniform(0.06, 0.1),
            **self._sample_spawn_field_updates(),
        )

    def _sample_spawn_parameters(
        self, params: BalloonParameters, seed: int, i: int
    ) -> BalloonParameters:
        return params.model_copy(update=self._sample_spawn_field_updates())

    def apply_parameters(
        self, params: BalloonParameters, *, spawn_scope: bool = True
    ) -> None:
        # NOTE: rel_scale sampled on self from seed; excluded from quartet sampling (uniform scale normalized away in point clouds).
        with FixedSeed(params.seed):
            self.rel_scale = uniform(0.2, 0.3) * 4
        self.thickness = params.thickness
        # NOTE: displace runs after the cloth sim and does not elicit a clear visual change in exported geometry; excluded from quartet sampling.
        with FixedSeed(params.seed):
            self.displace = uniform(0.02, 0.04)
        if not hasattr(self, "_material_gen"):
            self._material_gen, self._surface = self._sample_materials()
        self.material_gen = self._material_gen
        self.surface = self._surface
        self._use_fixed_spawn_draws = spawn_scope
        if spawn_scope:
            self.tension_stiffness = params.tension_stiffness
            self.uniform_pressure_force = params.uniform_pressure_force

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        bpy.ops.object.text_add()
        obj = bpy.context.active_object

        with butil.ViewportMode(obj, "EDIT"):
            for _ in "Text":
                bpy.ops.font.delete(type="PREVIOUS_OR_SELECTION")
            text = generate_text().upper()
            bpy.ops.font.text_insert(text=text)
        with butil.SelectObjects(obj):
            bpy.ops.object.convert(target="MESH")
        obj = bpy.context.active_object
        parent = new_bbox(
            -self.thickness / 2,
            self.thickness / 2,
            0,
            self.rel_scale * len(text) * self.alpha,
            0,
            self.rel_scale * self.alpha,
        )
        obj.parent = parent
        return parent

    def create_asset(self, i, placeholder, **params) -> bpy.types.Object:
        obj = placeholder.children[0]
        obj.parent = None
        remesh_fill(obj, 0.02)
        butil.modify_mesh(obj, "SOLIDIFY", thickness=self.thickness, offset=0.5)
        subdivide_edge_ring(obj, 8, (0, 0, 1))

        tension = (
            self.tension_stiffness
            if self._use_fixed_spawn_draws
            else uniform(0, 5)
        )
        pressure = (
            self.uniform_pressure_force
            if self._use_fixed_spawn_draws
            else uniform(10, 20)
        )
        clothes.cloth_sim(
            obj,
            tension_stiffness=tension,
            gravity=0,
            use_pressure=True,
            uniform_pressure_force=pressure,
            vertex_group_mass="pin",
        )

        subsurf(obj, 1)
        obj.scale = [self.rel_scale] * 3
        obj.rotation_euler = np.pi / 2, 0, np.pi / 2
        butil.apply_transform(obj, True)
        butil.modify_mesh(obj, "DISPLACE", strength=self.displace)
        butil.modify_mesh(obj, "SMOOTH", iterations=5)

        surface.assign_material(obj, self.surface)

        return obj
