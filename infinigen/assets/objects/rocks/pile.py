# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei

from __future__ import annotations

from typing import Annotated, ClassVar

import bpy
import numpy as np
import tqdm
from numpy.random import randint, uniform
from pydantic import Field

import infinigen.core.util.blender as butil
from infinigen.assets.objects.rocks.boulder import BoulderFactory
from infinigen.assets.utils.decorate import multi_res
from infinigen.assets.utils.draw import surface_from_func
from infinigen.assets.utils.misc import toggle_hide
from infinigen.assets.utils.object import join_objects
from infinigen.assets.utils.physics import free_fall
from infinigen.core.placement.detail import remesh_with_attrs
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import (
    AssetParameters,
    ParameterizedAssetFactory,
)
from infinigen.core.tagging import tag_object
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.random import log_uniform


class BoulderPileParameters(AssetParameters):
    boulder_count: Annotated[
        int, Field(ge=3, le=5, json_schema_extra={"editable": True})
    ]
    primary_scale: Annotated[
        float, Field(ge=0.4, le=0.6, json_schema_extra={"editable": True})
    ]
    secondary_scale: Annotated[
        float, Field(ge=0.2, le=0.4, json_schema_extra={"editable": True})
    ]
    tertiary_scale: Annotated[
        float, Field(ge=0.1, le=0.2, json_schema_extra={"editable": True})
    ]
    face_size: Annotated[
        float, Field(ge=0.005, le=0.02, json_schema_extra={"editable": True})
    ]
    floor_radius: Annotated[
        float, Field(ge=3.0, le=5.0, json_schema_extra={"editable": True})
    ]


class BoulderPileFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = BoulderPileParameters

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> BoulderPileParameters:
        return BoulderPileParameters(
            seed=seed,
            boulder_count=int(randint(3, 5)),
            primary_scale=float(log_uniform(0.4, 0.6)),
            secondary_scale=float(log_uniform(0.2, 0.4)),
            tertiary_scale=float(log_uniform(0.1, 0.2)),
            face_size=0.01,
            floor_radius=4.0,
        )

    def apply_parameters(
        self, params: BoulderPileParameters, *, spawn_scope: bool = True
    ) -> None:
        self._pile_params = params
        self.factory = BoulderFactory(self.factory_seed, coarse=self.coarse)
        self._use_fixed_spawn_draws = spawn_scope

    @staticmethod
    def create_floor(floor_radius: float):
        r = floor_radius

        def floor_fn(x, y):
            alpha = 0.01
            x = np.sqrt(x * x + y * y) - r
            return np.maximum(x, alpha * x)

        mesh = surface_from_func(floor_fn, 32, 32, 12, 12)
        obj = bpy.data.objects.new("floor", mesh)
        bpy.context.scene.collection.objects.link(obj)
        return obj

    @staticmethod
    def place_boulder(obj, height):
        obj.location = *uniform(-3, 3, 2), height
        obj.rotation_euler = 0, 0, uniform(0, np.pi * 2)
        return height + obj.dimensions[-1]

    def create_placeholder(self, **kwargs):
        params = self._pile_params if self._use_fixed_spawn_draws else None
        n = params.boulder_count if params is not None else int(randint(3, 5))
        primary_scale = (
            params.primary_scale
            if params is not None
            else float(log_uniform(0.4, 0.6))
        )
        secondary_scale = (
            params.secondary_scale
            if params is not None
            else float(log_uniform(0.2, 0.4))
        )
        tertiary_scale = (
            params.tertiary_scale
            if params is not None
            else float(log_uniform(0.1, 0.2))
        )
        floor_radius = params.floor_radius if params is not None else 4.0
        empty = butil.spawn_empty("placeholder", disp_type="CUBE", s=8)
        objects = []
        scale = [1, primary_scale, secondary_scale, secondary_scale, secondary_scale, tertiary_scale]
        for i in range(n):
            empty_ = butil.spawn_empty("placeholder", disp_type="CUBE", s=8)
            p = self.factory.create_placeholder()
            p.parent = empty_
            objects.append(p)
            for s in scale[1:]:
                p_ = butil.spawn_empty("placeholder", disp_type="CUBE", s=8)
                o = deep_clone_obj(p)
                o.scale = [s] * 3
                o.parent = p_
                p_.parent = empty_
                objects.append(o)
            empty_.parent = empty
        floor = self.create_floor(floor_radius)
        free_fall(objects, [floor], BoulderPileFactory.place_boulder)
        butil.delete(floor)
        return empty

    def create_asset(
        self, i: int, placeholder, face_size: float = 0.01, **params
    ) -> bpy.types.Object:
        if self._use_fixed_spawn_draws:
            face_size = self._pile_params.face_size
        objects = []
        for c in tqdm.tqdm(placeholder.children, desc="Creating boulder assets"):
            p = c.children[0]
            a = self.factory.create_asset(i=i, placeholder=p)
            a.location = p.location
            a.rotation_euler = p.rotation_euler
            objects.append(a)
            for clone_parent in c.children[1:]:
                clone_mesh = clone_parent.children[0]
                a_ = deep_clone_obj(a)
                a_.scale = clone_mesh.scale
                a_.location = clone_mesh.location
                a_.rotation_euler = clone_mesh.rotation_euler
                objects.append(a_)
                toggle_hide(clone_parent)
        obj = join_objects(objects)
        for c in placeholder.children:
            for p in c.children:
                butil.delete(p)
            butil.delete(c)
        multi_res(obj)
        remesh_with_attrs(obj, face_size)
        tag_object(obj, "pile")
        return obj
