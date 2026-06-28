# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


from typing import Annotated, ClassVar

import numpy as np
from mathutils import Euler, kdtree
from numpy.random import uniform
from pydantic import Field

from infinigen.assets.utils.mesh import polygon_angles
from infinigen.assets.utils.object import join_objects
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import (
    AssetParameters,
    ParameterizedAssetFactory,
)
from infinigen.core.tagging import tag_object
from infinigen.core.util import blender as butil
from infinigen.core.util.blender import deep_clone_obj

from .growth import MushroomGrowthFactory


class MushroomParameters(AssetParameters):
    maker_draw: Annotated[
        float,
        Field(ge=0.0, le=1.0, json_schema_extra={"editable": True, "kind": "draw_bool"}),
    ]
    lowered_draw: Annotated[
        float,
        Field(ge=0.0, le=1.0, json_schema_extra={"editable": True, "kind": "draw_bool"}),
    ]
    tolerant_length: Annotated[
        float, Field(ge=0.0, le=0.2, json_schema_extra={"editable": True})
    ]
    cluster_count: Annotated[
        int, Field(ge=1, le=5, json_schema_extra={"editable": False})
    ] = 1
    bend_angle: Annotated[
        float, Field(ge=-0.392699, le=0.392699, json_schema_extra={"editable": False})
    ] = 0.0
    bend_axis: Annotated[
        str,
        Field(
            json_schema_extra={
                "editable": False,
                "kind": "enum",
                "choices": ["X", "Y"],
            }
        ),
    ] = "X"


class MushroomFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = MushroomParameters
    max_cluster = 10

    def __init__(self, factory_seed, coarse=False):
        super(MushroomFactory, self).__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> MushroomParameters:
        self.makers = [self.directional_make, self.cluster_make]
        self.factory = MushroomGrowthFactory(seed, self.coarse)
        return MushroomParameters(
            seed=seed,
            maker_draw=uniform(0, 1),
            lowered_draw=uniform(0, 1),
            tolerant_length=uniform(0, 0.2),
        )

    def _sample_spawn_parameters(
        self, params: MushroomParameters, seed: int, i: int
    ) -> MushroomParameters:
        return params.model_copy(
            update={
                "cluster_count": int(np.random.randint(1, 6)),
                "bend_angle": uniform(-np.pi / 8, np.pi / 8),
                "bend_axis": str(np.random.choice(["X", "Y"])),
            }
        )

    def apply_parameters(
        self, params: MushroomParameters, *, spawn_scope: bool = True
    ) -> None:
        self.makers = [self.directional_make, self.cluster_make]
        self.maker = self.makers[0 if params.maker_draw < 0.5 else 1]
        self.lowered = params.lowered_draw < 0.5
        self.factory = MushroomGrowthFactory(params.seed, self.coarse)
        self.tolerant_length = params.tolerant_length
        self.cluster_count = params.cluster_count
        self.bend_angle = params.bend_angle
        self.bend_axis = params.bend_axis
        self._use_fixed_spawn_draws = spawn_scope


    def create_asset(self, i, face_size, **params):
        mushrooms, keypoints = self.build_mushrooms(i, face_size)
        locations, rotations, scales = self.maker(keypoints)
        for m, l, r, s in zip(mushrooms, locations, rotations, scales):
            m.location = l
            m.rotation_euler = r
            m.scale = s
            butil.apply_transform(m, loc=True)
        obj = join_objects(mushrooms)
        angle = (
            self.bend_angle
            if self._use_fixed_spawn_draws
            else uniform(-np.pi / 8, np.pi / 8)
        )
        axis = (
            self.bend_axis
            if self._use_fixed_spawn_draws
            else str(np.random.choice(["X", "Y"]))
        )
        butil.modify_mesh(
            obj,
            "SIMPLE_DEFORM",
            deform_method="BEND",
            angle=angle,
            deform_axis=axis,
        )
        tag_object(obj, "mushroom")
        return obj

    def build_mushrooms(self, i, face_size=0.01):
        n = self.cluster_count if self._use_fixed_spawn_draws else np.random.randint(1, 6)
        mushrooms, keypoints = [], []
        for j in range(n):
            obj = self.factory.create_asset(
                i=j + i * self.max_cluster, face_size=face_size / 2
            )
            clone = deep_clone_obj(obj)
            butil.modify_mesh(clone, "REMESH", voxel_size=0.04)
            mushrooms.append(obj)
            k = np.array(
                [v.co for v in clone.data.vertices if v.co[-1] > self.tolerant_length]
            )
            if len(k) == 0:
                k = np.array([v.co for v in clone.data.vertices])
            if len(k) == 0:
                k = np.zeros((1, 3))
            keypoints.append(k)
            butil.delete(clone)
        return mushrooms, keypoints

    @property
    def radius(self):
        return self.factory.cap_factory.radius

    def find_closest(self, keypoints, rotations, start_locs, directions):
        vertices = [k.copy() for k in keypoints]
        locations, scales = [np.zeros(3)], []
        scales = np.tile(uniform(0.3, 1.2, len(keypoints))[:, np.newaxis], 3)
        for i in range(len(vertices)):
            vertices[i] = (
                np.array(Euler(rotations[i]).to_matrix())
                @ np.diag(scales[i])
                @ vertices[i].T
            ).T
        for i in range(1, len(vertices)):
            basis = np.concatenate(vertices[:i])
            kd = kdtree.KDTree(len(basis))
            for idx, v in enumerate(basis):
                kd.insert(v, idx)
            kd.balance()
            for d in np.linspace(0, 4, 20) * self.radius:
                offset = start_locs[i] + directions[i] * d
                if min(kd.find(v + offset)[-1] for v in vertices[i]) > 0.008:
                    break
            else:
                offset = start_locs[i] + directions[i] * 4 * self.radius
            vertices[i] += offset
            locations.append(offset)
        return locations, rotations, scales

    def cluster_make(self, keypoints):
        n = len(keypoints)
        angles = polygon_angles(n, np.pi / 10, np.pi * 2)
        rot_y = uniform(0, np.pi / 6, n) if self.lowered else np.zeros(n)
        rot_z = angles + uniform(-np.pi / 8, np.pi / 8, n)
        rotations = np.stack([np.zeros(n), rot_y, rot_z], -1)
        start_locs = np.zeros((n, 3))
        directions = np.stack([np.cos(angles), np.sin(angles), np.zeros(n)], -1)
        return self.find_closest(keypoints, rotations, start_locs, directions)

    def directional_make(self, keypoints):
        n = len(keypoints)
        rot_y = uniform(0, np.pi / 6, n) if self.lowered else np.zeros(n)
        rot_z = -np.pi / 2 + uniform(-np.pi / 8, np.pi / 8, n)
        rotations = np.stack([np.zeros(n), rot_y, rot_z], -1)
        start_locs = np.stack(
            [np.linspace(0, self.radius * n * 0.4, n), np.zeros(n), np.zeros(n)], -1
        )
        directions = np.tile([0, 1, 0], (n, 1))
        return self.find_closest(keypoints, rotations, start_locs, directions)
