# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Lingjie Mei


from __future__ import annotations

from typing import Annotated, ClassVar

import bpy
import numpy as np
from numpy.random import uniform
from pydantic import Field

from infinigen.assets.objects.deformed_trees.base import (
    BaseDeformedTreeFactory,
    BaseDeformedTreeParameters,
)
from infinigen.assets.utils.decorate import remove_vertices
from infinigen.assets.utils.draw import cut_plane
from infinigen.assets.utils.misc import assign_material
from infinigen.assets.utils.object import join_objects, separate_loose
from infinigen.core import surface
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.placement.parameters import AssetParameters
from infinigen.core.tagging import tag_object
from infinigen.core.util import blender as butil
from infinigen.core.util.blender import deep_clone_obj


class FallenTreeParameters(BaseDeformedTreeParameters):
    cut_center_z: Annotated[
        float, Field(ge=0.6, le=1.2, json_schema_extra={"editable": True})
    ]
    cut_normal_x: Annotated[
        float, Field(ge=0.1, le=0.2, json_schema_extra={"editable": True})
    ]
    noise_strength: Annotated[
        float, Field(ge=0.3, le=0.5, json_schema_extra={"editable": True})
    ]
    noise_scale: Annotated[
        float, Field(ge=10.0, le=15.0, json_schema_extra={"editable": True})
    ]
    fall_offset_x: Annotated[
        float, Field(ge=0.05, le=0.15, json_schema_extra={"editable": False})
    ] = 0.1
    fall_offset_z: Annotated[
        float, Field(ge=0.05, le=0.15, json_schema_extra={"editable": False})
    ] = 0.1
    tilt_height: Annotated[
        float, Field(ge=0.0, le=0.2, json_schema_extra={"editable": False})
    ] = 0.1


class FallenTreeFactory(BaseDeformedTreeFactory):
    parameters_model: ClassVar[type[AssetParameters]] = FallenTreeParameters

    def _sample_init_parameters(self, seed: int) -> FallenTreeParameters:
        base = super()._sample_init_parameters(seed)
        return FallenTreeParameters(
            **base.model_dump(),
            cut_center_z=uniform(0.6, 1.2),
            cut_normal_x=uniform(0.1, 0.2),
            noise_strength=uniform(0.3, 0.5),
            noise_scale=uniform(10, 15),
        )

    def _sample_spawn_parameters(
        self, params: FallenTreeParameters, seed: int, i: int
    ) -> FallenTreeParameters:
        return params.model_copy(
            update={
                "fall_offset_x": uniform(0.05, 0.15),
                "fall_offset_z": uniform(0.05, 0.15),
                "tilt_height": uniform(0, 0.2),
            }
        )

    def apply_parameters(
        self, params: FallenTreeParameters, *, spawn_scope: bool = True
    ) -> None:
        super().apply_parameters(params, spawn_scope=spawn_scope)
        self.cut_center_z = params.cut_center_z
        self.cut_normal_x = params.cut_normal_x
        self.noise_strength = params.noise_strength
        self.noise_scale = params.noise_scale
        self._fall_offset_x = params.fall_offset_x
        self._fall_offset_z = params.fall_offset_z
        self._tilt_height = params.tilt_height

    @staticmethod
    def geo_cutter(nw: NodeWrangler, strength, scale, radius, metric_fn):
        geometry = nw.new_node(
            Nodes.GroupInput, expose_input=[("NodeSocketGeometry", "Geometry", None)]
        )
        x, y, z = nw.separate(nw.new_node(Nodes.InputPosition))
        selection = nw.compare(
            "LESS_THAN", nw.scalar_add(nw.power(x, 2), nw.power(y, 2)), 1
        )
        offset = nw.scalar_multiply(
            nw.new_node(
                Nodes.Clamp,
                [
                    nw.new_node(
                        Nodes.NoiseTexture,
                        input_kwargs={
                            "Vector": nw.new_node(Nodes.InputPosition),
                            "Scale": scale,
                        },
                    ),
                    0.3,
                    0.7,
                ],
            ),
            strength,
        )
        offset = nw.scalar_multiply(
            offset, nw.build_float_curve(x, [(-radius, 1), (radius, 0)])
        )
        anchors = (-1, 0), (-0.5, 0), (0, -1), (0.5, 0), (1, 0)
        offset = nw.scalar_multiply(
            offset, nw.build_float_curve(surface.eval_argument(nw, metric_fn), anchors)
        )
        geometry = nw.new_node(
            Nodes.SetPosition, [geometry, selection, None, nw.combine(0, 0, offset)]
        )
        nw.new_node(Nodes.GroupOutput, input_kwargs={"Geometry": geometry})

    def build_half(
        self,
        obj,
        cut_center,
        cut_normal,
        noise_strength,
        noise_scale,
        radius,
        is_up=True,
    ):
        obj, cut = cut_plane(obj, cut_center, cut_normal, not is_up)
        assign_material(cut, self.material)
        obj = join_objects([obj, cut])
        with butil.ViewportMode(obj, "EDIT"), butil.Suppress():
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.region_to_loop()
            bpy.ops.mesh.remove_doubles(threshold=1e-2)
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.fill_holes()

        def metric_fn(nw):
            return nw.dot(
                nw.sub(nw.new_node(Nodes.InputPosition), cut_center), cut_normal
            )

        surface.add_geomod(
            obj,
            self.geo_cutter,
            apply=True,
            input_args=[noise_strength, noise_scale, radius, metric_fn],
        )
        obj = separate_loose(obj)
        surface.add_geomod(obj, self.geo_xyz, apply=True)
        return obj

    def create_asset(self, i, distance=0, **params):
        upper = self.build_tree(i, distance, **params)
        radius = max(
            [
                np.sqrt(v.co[0] ** 2 + v.co[1] ** 2)
                for v in upper.data.vertices
                if v.co[-1] < 0.1
            ]
        )
        self.trunk_surface.apply(upper)
        butil.apply_modifiers(upper)
        lower = deep_clone_obj(upper, keep_materials=True)
        cut_center = np.array([0, 0, self.cut_center_z])
        cut_normal = np.array([self.cut_normal_x, 0, 1])
        upper = self.build_half(
            upper,
            cut_center,
            cut_normal,
            self.noise_strength,
            self.noise_scale,
            radius,
            True,
        )
        lower = self.build_half(
            lower,
            cut_center,
            cut_normal,
            self.noise_strength,
            self.noise_scale,
            radius,
            False,
        )

        ortho = np.array([-cut_normal[0], 0, 1])
        locations = np.array([v.co for v in lower.data.vertices])
        if self._use_fixed_spawn_draws:
            offset_x = self._fall_offset_x
            offset_z = self._fall_offset_z
            tilt_height = self._tilt_height
        else:
            offset_x = uniform(0.05, 0.15)
            offset_z = uniform(0.05, 0.15)
            tilt_height = uniform(0, 0.2)
        highest = locations[np.argmax(locations @ ortho)] + np.array(
            [-offset_x, 0, -offset_z]
        )
        upper.location = -highest
        butil.apply_transform(upper, loc=True)

        x, _, z = np.mean(np.stack([v.co for v in upper.data.vertices]), 0)
        r = np.sqrt(x * x + z * z)
        if r > 0:
            upper.rotation_euler[1] = (
                np.pi / 2
                + np.arcsin((highest[-1] - tilt_height) / r)
                - np.arctan(x / z)
            )
        upper.location = highest
        butil.apply_transform(upper, loc=True)
        remove_vertices(upper, lambda x, y, z: z < -0.5)
        upper = separate_loose(upper)
        obj = join_objects([upper, lower])
        tag_object(obj, "fallen_tree")
        return obj
