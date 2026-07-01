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
from infinigen.assets.utils.decorate import (
    read_co,
    read_edge_center,
    read_selected,
    select_edges,
    subdivide_edge_ring,
    subsurf,
    write_co,
)
from infinigen.assets.utils.object import (
    join_objects,
    new_base_circle,
    new_cylinder,
)
from infinigen.core import surface
from infinigen.core.constraints.constraint_language.constants import RoomConstants
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import (
    AssetParameters,
    ParameterizedAssetFactory,
)
from infinigen.core.util import blender as butil
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform, weighted_sample


class PillarParameters(AssetParameters):
    detail_type: Annotated[
        str,
        Field(
            json_schema_extra={
                "editable": True,
                "kind": "enum",
                "choices": ["fluting", "reeding"],
            }
        ),
    ] = "fluting"


def _pillar_legacy_init(
    inst: AssetFactory,
    seed: int,
    coarse: bool,
    constants: RoomConstants | None,
) -> None:
        with FixedSeed(seed):
            if constants is None:
                constants = RoomConstants()
            inst.height = constants.wall_height - constants.wall_thickness
            # NOTE: n, lower_offset, upper_offset do not elicit a clear visual change in exported geometry; excluded from quartet sampling.
            inst.n = int(np.random.randint(5, 10))
            # NOTE: radius, outer_radius_ratio, and inset_depth do not elicit a clear visual change in exported geometry; excluded from quartet sampling.
            inst.radius = uniform(0.08, 0.12)
            inst.outer_radius = inst.radius * uniform(1.3, 1.5)
            inst.lower_offset = uniform(0.05, 0.15)
            inst.upper_offset = uniform(0.05, 0.15)
            inst.detail_type = np.random.choice(["fluting", "reeding"])
        width = np.pi / 2 / inst.n
        inst.inset_width = width * log_uniform(0.1, 0.2)
        inst.inset_width_ = (width - inst.inset_width * 2) * uniform(-0.1, 0.3)
        inst.inset_depth = uniform(0.1, 0.15)
        inst.inset_scale = uniform(0.05, 0.1)
        inst.outer_n = np.random.choice([1, 2, inst.n])
        inst.m = np.random.randint(12, 20)
        z_profile = uniform(1, 3, inst.m)
        inst.z_profile = np.array(
            [0, *(np.cumsum(z_profile) / np.sum(z_profile))[:-1]]
        )
        alpha = uniform(0.7, 0.85)
        r_profile = uniform(0, 1, inst.m + 3)
        r_profile[[0, 1]] = 1
        r_profile[[-2, -1]] = 0
        r_profile = np.convolve(
            r_profile, np.array([(1 - alpha) / 2, alpha, (1 - alpha) / 2])
        )
        inst.r_profile = (
            np.array([1, *r_profile[2:-2]]) * (inst.outer_radius - inst.radius)
            + inst.radius
        )
        inst.n_profile = np.where(
            np.arange(inst.m) < np.random.randint(2, inst.m - 1),
            inst.outer_n,
            inst.n,
        )
        inst.inset_profile = uniform(0, 1, inst.m) < 0.3
        inst.surface = weighted_sample(material_assignments.marble)()


class PillarFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = PillarParameters

    def __init__(self, factory_seed, coarse=False, constants=None):
        self._constants_arg = constants
        super().__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> PillarParameters:
        with FixedSeed(seed):
            return PillarParameters(
                seed=seed,
                detail_type="fluting",
            )

    def apply_parameters(
        self, params: PillarParameters, *, spawn_scope: bool = True
    ) -> None:
        _pillar_legacy_init(
            self, params.seed, self.coarse, self._constants_arg
        )
        self.detail_type = params.detail_type
        self._use_fixed_spawn_draws = spawn_scope

    def create_asset(self, **params) -> bpy.types.Object:
        obj = new_cylinder(vertices=4 * self.n)
        with butil.ViewportMode(obj, "EDIT"):
            bm = bmesh.from_edit_mesh(obj.data)
            geom = [f for f in bm.faces if len(f.verts) > 4]
            bmesh.ops.delete(bm, geom=geom, context="FACES_ONLY")
            bmesh.update_edit_mesh(obj.data)

        obj.scale = (
            self.radius,
            self.radius,
            (1 - self.lower_offset - self.upper_offset) * self.height,
        )
        obj.location[-1] = self.lower_offset * self.height
        butil.apply_transform(obj, True)
        inset_scale = 1 + self.inset_scale * (
            1 if self.detail_type == "reeding" else -1
        )
        if self.detail_type in ["fluting", "reeding"]:
            with butil.ViewportMode(obj, "EDIT"):
                bpy.ops.mesh.select_mode(type="FACE")
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.inset(
                    thickness=self.inset_width * self.radius, use_individual=True
                )
                bpy.ops.mesh.inset(
                    thickness=self.inset_width_ * self.radius, use_individual=True
                )
                bpy.ops.transform.resize(value=(inset_scale, inset_scale, 1))
        subdivide_edge_ring(obj, 16)
        parts = [obj]
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_mode(type="EDGE")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.region_to_loop()
        z_rot = np.pi / 2 * np.random.randint(2)
        for z, r, n, i in zip(
            self.z_profile, self.r_profile, self.n_profile, self.inset_profile
        ):
            o = new_base_circle(vertices=4 * n)
            if i:
                co = read_co(o)
                stride = np.random.choice([2, 4, 8])
                co *= np.where(np.arange(len(co)) % stride == 0, 1, inset_scale)[
                    :, np.newaxis
                ]
                write_co(o, co)
            with butil.ViewportMode(o, "EDIT"):
                bpy.ops.mesh.select_mode(type="EDGE")
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.subdivide(number_cuts=self.n // n - 1)
            o.location[-1] = z * self.lower_offset * self.height
            r_ = r / np.cos(np.pi / 4 / n)
            o.scale = r_, r_, 1
            o.rotation_euler[-1] = z_rot
            o_ = deep_clone_obj(o)
            o_.location[-1] = (1 - z * self.upper_offset) * self.height
            butil.apply_transform(o, True)
            butil.apply_transform(o_, True)
            parts.extend([o, o_])
        obj = join_objects(parts)
        selection = read_selected(obj, "EDGE")
        z = read_edge_center(obj)[:, -1]
        number_cuts = 0
        smoothness = uniform(1, 1.4)
        select_edges(obj, selection & (z < 0.5))
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.bridge_edge_loops(
                number_cuts=number_cuts, smoothness=smoothness
            )
        select_edges(obj, selection & (z > 0.5))
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.bridge_edge_loops(
                number_cuts=number_cuts, smoothness=smoothness
            )
        subsurf(obj, 1, True)
        subsurf(obj, 1)
        return obj

    def finalize_assets(self, assets):
        surface.assign_material(assets, self.surface())
