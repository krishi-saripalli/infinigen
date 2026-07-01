# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Hei Law, Alexander Raistrick

# ruff: noqa: I001

from __future__ import annotations

from typing import Annotated, ClassVar

import bpy
from numpy.random import normal as N, randint, uniform
from pydantic import Field

from infinigen.assets.materials.terrain import dirt
from infinigen.core import surface
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import AssetParameters, ParameterizedAssetFactory
from infinigen.core.tagging import tag_object
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.infinigen_gpl.surfaces import snow


def shader_raindrop(nw):
    glass_bsdf = nw.new_node(
        "ShaderNodeBsdfGlass",
        input_kwargs={
            "IOR": 1.33,
        },
    )
    material_output = nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={
            "Surface": glass_bsdf,
        },
    )


def geo_raindrop(nw, curve_depth: float = 0.15):
    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            (
                "NodeSocketGeometry",
                "Geometry",
                None,
            )
        ],
    )

    position = nw.new_node(Nodes.InputPosition)

    vector_curves = nw.new_node(
        Nodes.VectorCurve,
        input_kwargs={
            "Vector": position,
        },
    )
    node_utils.assign_curve(
        vector_curves.mapping.curves[0],
        [(-1.0, -1.0), (1.0, 1.0)],
    )
    node_utils.assign_curve(
        vector_curves.mapping.curves[1],
        [(-1.0, -1.0), (1.0, 1.0)],
    )
    node_utils.assign_curve(
        vector_curves.mapping.curves[2],
        [(-1.0, -curve_depth * N(1, 0.15)), (-0.6091, -0.0938), (1.0, 1.0)],
    )

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": group_input.outputs["Geometry"],
            "Position": vector_curves,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Geometry": set_position,
        },
    )


class RaindropParameters(AssetParameters):
    radius: Annotated[
        float, Field(ge=0.5, le=2.0, json_schema_extra={"editable": False})
    ] = 1.0
    subdivisions: Annotated[
        int, Field(ge=3, le=6, json_schema_extra={"editable": False})
    ] = 5
    curve_depth: Annotated[
        float, Field(ge=0.08, le=0.22, json_schema_extra={"editable": True})
    ] = 0.15
    scale_x: Annotated[
        float, Field(ge=0.7, le=1.3, json_schema_extra={"editable": False})
    ] = 1.0
    scale_y: Annotated[
        float, Field(ge=0.7, le=1.3, json_schema_extra={"editable": False})
    ] = 1.0
    scale_z: Annotated[
        float, Field(ge=0.7, le=1.3, json_schema_extra={"editable": False})
    ] = 1.0


class RaindropFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = RaindropParameters

    def __init__(self, factory_seed=None, coarse=False):
        super().__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> RaindropParameters:
        return RaindropParameters(seed=seed)

    def _sample_spawn_parameters(
        self, params: RaindropParameters, seed: int, i: int
    ) -> RaindropParameters:
        return params.model_copy(
            update={
                "radius": uniform(0.5, 2.0),
                "subdivisions": int(randint(3, 7)),
                "curve_depth": uniform(0.08, 0.22),
                "scale_x": uniform(0.7, 1.3),
                "scale_y": uniform(0.7, 1.3),
                "scale_z": uniform(0.7, 1.3),
            }
        )

    def apply_parameters(
        self, params: RaindropParameters, *, spawn_scope: bool = True
    ) -> None:
        self._use_fixed_spawn_draws = spawn_scope
        if spawn_scope:
            self._raindrop_params = params

    def create_asset(self, **kwargs):
        if self._use_fixed_spawn_draws:
            params = self._raindrop_params
            radius = params.radius
            subdivisions = params.subdivisions
            curve_depth = params.curve_depth
            scale = (params.scale_x, params.scale_y, params.scale_z)
        else:
            radius = uniform(0.5, 2.0)
            subdivisions = int(randint(3, 7))
            curve_depth = uniform(0.08, 0.22)
            scale = (uniform(0.7, 1.3), uniform(0.7, 1.3), uniform(0.7, 1.3))
        bpy.ops.mesh.primitive_ico_sphere_add(
            radius=radius,
            enter_editmode=False,
            subdivisions=subdivisions,
            align="WORLD",
            location=(0, 0, 0),
            scale=scale,
        )

        sphere = bpy.context.object

        surface.add_geomod(
            sphere, geo_raindrop, apply=True, input_kwargs={"curve_depth": curve_depth}
        )
        tag_object(sphere, "raindrop")
        return sphere

    def finalize_assets(self, assets):
        surface.add_material(assets, shader_raindrop)


class DustMoteParameters(AssetParameters):
    subdivisions: Annotated[
        int, Field(ge=1, le=4, json_schema_extra={"editable": False})
    ] = 2


class DustMoteFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = DustMoteParameters

    def __init__(self, factory_seed=None, coarse=False):
        super().__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> DustMoteParameters:
        return DustMoteParameters(seed=seed)

    def _sample_spawn_parameters(
        self, params: DustMoteParameters, seed: int, i: int
    ) -> DustMoteParameters:
        return params.model_copy(
            update={
                "subdivisions": int(randint(1, 5)),
            }
        )

    def apply_parameters(
        self, params: DustMoteParameters, *, spawn_scope: bool = True
    ) -> None:
        self._use_fixed_spawn_draws = spawn_scope
        if spawn_scope:
            self._subdivisions = params.subdivisions
            with FixedSeed(params.seed):
                self._radius = uniform(0.3, 2.0)
                self._scale = uniform(0.5, 2.0)

    def create_asset(self, **kwargs):
        radius = self._radius if self._use_fixed_spawn_draws else uniform(0.3, 2.0)
        subdivisions = (
            self._subdivisions if self._use_fixed_spawn_draws else int(randint(1, 5))
        )
        scale = self._scale if self._use_fixed_spawn_draws else uniform(0.5, 2.0)
        bpy.ops.mesh.primitive_ico_sphere_add(
            radius=radius,
            subdivisions=subdivisions,
            enter_editmode=False,
            align="WORLD",
            location=(0, 0, 0),
            scale=(scale, scale, scale),
        )
        tag_object(bpy.context.object, "dustmote")
        return bpy.context.object

    def finalize_assets(self, assets):
        dirt.apply(assets)


class SnowflakeParameters(AssetParameters):
    radius: Annotated[
        float, Field(ge=0.3, le=2.0, json_schema_extra={"editable": False})
    ] = 1.0
    vertices: Annotated[
        int, Field(ge=6, le=12, json_schema_extra={"editable": True})
    ] = 6
    scale_x: Annotated[
        float, Field(ge=0.5, le=2.0, json_schema_extra={"editable": False})
    ] = 1.0
    scale_y: Annotated[
        float, Field(ge=0.5, le=2.0, json_schema_extra={"editable": False})
    ] = 1.0
    rotation_z: Annotated[
        float, Field(ge=0.0, le=6.283185, json_schema_extra={"editable": False})
    ] = 0.0


class SnowflakeFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = SnowflakeParameters

    def __init__(self, factory_seed=None, coarse=False):
        super().__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> SnowflakeParameters:
        return SnowflakeParameters(seed=seed)

    def _sample_spawn_parameters(
        self, params: SnowflakeParameters, seed: int, i: int
    ) -> SnowflakeParameters:
        return params.model_copy(
            update={
                "radius": uniform(0.3, 2.0),
                "vertices": int(randint(6, 13)),
                "scale_x": uniform(0.5, 2.0),
                "scale_y": uniform(0.5, 2.0),
                "rotation_z": uniform(0.0, 6.283185),
            }
        )

    def apply_parameters(
        self, params: SnowflakeParameters, *, spawn_scope: bool = True
    ) -> None:
        self._use_fixed_spawn_draws = spawn_scope
        if spawn_scope:
            self._radius = params.radius
            self._vertices = params.vertices
            self._scale_x = params.scale_x
            self._scale_y = params.scale_y
            self._rotation_z = params.rotation_z

    def create_asset(self, **params) -> bpy.types.Object:
        radius = self._radius if self._use_fixed_spawn_draws else uniform(0.3, 2.0)
        vertices = (
            self._vertices if self._use_fixed_spawn_draws else int(randint(6, 13))
        )
        scale_x = self._scale_x if self._use_fixed_spawn_draws else uniform(0.5, 2.0)
        scale_y = self._scale_y if self._use_fixed_spawn_draws else uniform(0.5, 2.0)
        rotation_z = (
            self._rotation_z if self._use_fixed_spawn_draws else uniform(0.0, 6.283185)
        )
        bpy.ops.mesh.primitive_circle_add(
            vertices=vertices,
            fill_type="TRIFAN",
            radius=radius,
        )
        obj = bpy.context.object
        obj.scale = (scale_x, scale_y, 1.0)
        obj.rotation_euler[2] = rotation_z
        butil.apply_transform(obj)
        tag_object(obj, "snowflake")
        return obj

    def finalize_assets(self, assets):
        snow.apply(assets, subsurface=0)
