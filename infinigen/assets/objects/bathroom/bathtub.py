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
from infinigen.assets.utils.autobevel import BevelSharp
from infinigen.assets.utils.decorate import (
    read_center,
    read_co,
    read_normal,
    subsurf,
    write_attribute,
    write_co,
)
from infinigen.assets.utils.nodegroup import geo_radius
from infinigen.assets.utils.object import (
    join_objects,
    new_bbox,
    new_cube,
    new_cylinder,
    new_line,
)
from infinigen.core import surface
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import (
    AssetParameters,
    ParameterizedAssetFactory,
)
from infinigen.core.util import blender as butil
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import weighted_sample


class BathtubParameters(AssetParameters):
    width: Annotated[float, Field(ge=1.5, le=2.0, json_schema_extra={"editable": True})]
    size: Annotated[float, Field(ge=0.8, le=1.0, json_schema_extra={"editable": True})]
    depth: Annotated[float, Field(ge=0.55, le=0.7, json_schema_extra={"editable": True})]
    bathtub_type: Annotated[
        str,
        Field(
            json_schema_extra={
                "editable": False,
                "kind": "enum",
                "choices": ["alcove", "freestanding"],
            }
        ),
    ] = "alcove"
    has_curve: Annotated[
        bool, Field(json_schema_extra={"editable": False, "kind": "bool"})
    ]
    has_legs: Annotated[
        bool, Field(json_schema_extra={"editable": False, "kind": "bool"})
    ] = True
    leg_height_ratio: Annotated[
        float,
        Field(
            ge=0.2,
            le=0.3,
            json_schema_extra={"editable": False},
        ),
    ]
    leg_side: Annotated[
        float,
        Field(
            ge=0.05,
            le=0.1,
            json_schema_extra={"editable": False},
        ),
    ]
    leg_radius: Annotated[
        float,
        Field(
            ge=0.02,
            le=0.03,
            json_schema_extra={"editable": False},
        ),
    ]
    leg_y_scale_draw: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            json_schema_extra={"editable": False},
        ),
    ]
    leg_subsurf_level: Annotated[
        int,
        Field(
            ge=0,
            le=2,
            json_schema_extra={"editable": False},
        ),
    ]
    taper_factor: Annotated[
        float,
        Field(
            ge=-0.1,
            le=0.1,
            json_schema_extra={"editable": False},
        ),
    ]
    stretch_factor: Annotated[
        float,
        Field(
            ge=-0.2,
            le=0.2,
            json_schema_extra={"editable": False},
        ),
    ]
    leg_bevel_factor: Annotated[
        float,
        Field(
            ge=0.3,
            le=0.7,
            json_schema_extra={"editable": False},
        ),
    ] = 0.5
    freestanding_z_factor: Annotated[
        float,
        Field(
            ge=0.5,
            le=0.7,
            json_schema_extra={"editable": False},
        ),
    ] = 0.6
    hole_x_ratio: Annotated[
        float,
        Field(
            ge=0.35,
            le=0.4,
            json_schema_extra={"editable": False},
        ),
    ] = 0.375


def _init_bathtub_excluded(inst: BathtubFactory, seed: int, bathtub_type: str) -> None:
    inst.bathtub_type = bathtub_type
    with FixedSeed(seed):
        inst.contour_fn = (
            inst.make_corner_contour if inst.has_corner else inst.make_box_contour
        )
        inst.alcove_levels = np.random.randint(1, 3) if inst.has_base else 1
        inst.thickness = (
            uniform(0.04, 0.08) if inst.has_base else uniform(0.02, 0.04)
        )
        inst.surface_material_gen = weighted_sample(material_assignments.ceramics)
        inst.leg_surface_material_gen = weighted_sample(material_assignments.metal_neutral)
        inst.hole_surface_material_gen = weighted_sample(material_assignments.metal_neutral)
        inst.beveler = BevelSharp(mult=5, segments=5)
        inst.levels = 5
        inst.side_levels = 2
        inst.is_hole_centered = False


class BathtubFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = BathtubParameters

    def __init__(self, factory_seed, coarse=False):
        super(BathtubFactory, self).__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> BathtubParameters:
        return BathtubParameters(
            seed=seed,
            width=uniform(1.5, 2),
            size=uniform(0.8, 1),
            depth=uniform(0.55, 0.7),
            bathtub_type=str(np.random.choice(["alcove", "freestanding"])),
            has_curve=bool(uniform() < 0.5),
            has_legs=True,
            leg_height_ratio=uniform(0.2, 0.3),
            leg_side=uniform(0.05, 0.1),
            leg_radius=uniform(0.02, 0.03),
            leg_y_scale_draw=uniform(),
            leg_subsurf_level=int(np.random.randint(0, 3)),
            leg_bevel_factor=uniform(0.3, 0.7),
            freestanding_z_factor=uniform(0.5, 0.7),
            hole_x_ratio=uniform(0.35, 0.4),
            taper_factor=uniform(-0.1, 0.1),
            stretch_factor=uniform(-0.2, 0.2),
        )

    def _sample_spawn_parameters(
        self, params: BathtubParameters, seed: int, i: int
    ) -> BathtubParameters:
        return params

    def apply_parameters(
        self, params: BathtubParameters, *, spawn_scope: bool = True
    ) -> None:
        _init_bathtub_excluded(self, params.seed, params.bathtub_type)
        scratch_prob, edge_wear_prob = material_assignments.wear_tear_prob
        scratch_fn, edge_wear_fn = material_assignments.wear_tear
        self.width = params.width
        self.size = params.size
        self.depth = params.depth
        # NOTE: disp_x0/disp_x1/disp_y sampled on self from seed; excluded from quartet sampling.
        # make_bowl/make_cutter loft the lower rim contour to an upper rim
        # contour offset in the *opposite* direction (make_box_contour's `i`
        # sign flip). Even a small nonzero offset here combined with
        # alcove_levels' subsurf smoothing can make the boolean cutter
        # self-intersect at the rim, corrupting the cavity floor's normals
        # (renders as a black fan/streak inside the tub) -- confirmed by
        # bisecting down from the original absolute uniform(0, 0.2)/(0, 0.1)
        # draws: 1% and even 0.5% of width/size still broke the worst-case
        # seed in this dataset, only 0 was reliably clean across a full
        # 8-seed x 3-param sweep. This purely cosmetic asymmetry isn't worth
        # the boolean-robustness risk, so it's disabled.
        self.disp_x0 = 0.0
        self.disp_x1 = 0.0
        self.disp_y = 0.0
        self.disp_x = np.array([self.disp_x0, self.disp_x1])
        self.has_curve = params.has_curve
        self.has_legs = params.has_legs
        self.leg_height = params.leg_height_ratio * params.depth
        self.leg_side = params.leg_side
        self.leg_radius = params.leg_radius
        self.leg_y_scale = params.leg_y_scale_draw
        self.leg_subsurf_level = params.leg_subsurf_level
        self.taper_factor = params.taper_factor
        self.stretch_factor = params.stretch_factor
        # NOTE: hole_radius sampled on self from seed; excluded from quartet sampling (uniform scale normalized away in point clouds).
        with FixedSeed(params.seed):
            self.hole_radius = uniform(0.015, 0.02)
            scratch_draw = uniform()
            edge_wear_draw = uniform()
        self.scratch = (
            None if scratch_draw > scratch_prob else scratch_fn()
        )
        self.edge_wear = (
            None if edge_wear_draw > edge_wear_prob else edge_wear_fn()
        )
        self.leg_bevel_factor = params.leg_bevel_factor
        self.freestanding_z_factor = params.freestanding_z_factor
        self.hole_x_ratio = params.hole_x_ratio
        self._use_fixed_spawn_draws = spawn_scope

    @property
    def has_base(self):
        return self.bathtub_type != "freestanding"

    @property
    def has_corner(self):
        return self.bathtub_type == "corner"

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        return new_bbox(-self.size, 0, 0, self.width, 0, self.depth)

    def create_asset(self, **params) -> bpy.types.Object:
        self.surface = self.surface_material_gen()
        self.leg_surface = self.leg_surface_material_gen()
        self.hole_surface = self.hole_surface_material_gen()

        if self.has_base:
            obj = self.make_base()
            cutter = self.make_cutter()
            butil.modify_mesh(obj, "BOOLEAN", object=cutter, operation="DIFFERENCE")
            butil.delete(cutter)
            # The boolean occasionally leaves the cavity floor face with an
            # inward-facing normal (renders as a black n-gon fan inside the
            # tub) -- same fixup add_base() already applies after its own
            # boolean, just missing here.
            with butil.ViewportMode(obj, "EDIT"):
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.normals_make_consistent(inside=False)
        else:
            obj = self.make_freestanding()
            parts = [obj]
            if self.has_legs:
                parts.extend(self.make_legs(obj))
            else:
                parts.append(self.add_base(obj))
            butil.modify_mesh(obj, "SOLIDIFY", thickness=self.thickness)
            subsurf(obj, self.side_levels)
            obj = join_objects(parts)
        hole = self.add_hole(obj)
        obj = join_objects([obj, hole])
        obj.rotation_euler[-1] = np.pi / 2
        butil.apply_transform(obj, True)

        if self.bathtub_type == "freestanding":
            butil.modify_mesh(obj, "SUBSURF", levels=1, apply=True)
        else:
            self.beveler(obj)

        return obj

    def make_freestanding(self):
        obj = self.make_bowl()
        self.remove_top(obj)
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_mode(type="EDGE")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.region_to_loop()
            bpy.ops.mesh.extrude_edges_move()
            bpy.ops.transform.resize(
                value=(
                    1 + self.thickness * 2 / self.width,
                    1 + self.thickness / self.size,
                    1,
                )
            )
        obj.location[1] -= self.size / 2
        butil.apply_transform(obj, True)
        butil.modify_mesh(
            obj, "SIMPLE_DEFORM", deform_method="TAPER", angle=self.taper_factor
        )
        butil.modify_mesh(
            obj, "SIMPLE_DEFORM", deform_method="STRETCH", angle=self.stretch_factor
        )
        z_factor = (
            self.freestanding_z_factor
            if self._use_fixed_spawn_draws
            else uniform(0.5, 0.7)
        )
        obj.location = (
            0,
            self.size / 2,
            -np.min(read_co(obj)[:, -1]) * z_factor,
        )
        butil.apply_transform(obj, True)
        return obj

    def remove_top(self, obj):
        butil.select_none()
        with butil.ViewportMode(obj, "EDIT"):
            bm = bmesh.from_edit_mesh(obj.data)
            geom = [f for f in bm.faces if f.calc_center_median()[-1] > self.depth]
            bmesh.ops.delete(bm, geom=geom, context="FACES_KEEP_BOUNDARY")
            bmesh.update_edit_mesh(obj.data)

    def make_legs(self, obj):
        legs = []
        co, normal = read_center(obj), read_normal(obj)
        x, y, z = co.T
        leg_height = np.min(z) + self.leg_height
        bevel_factor = (
            self.leg_bevel_factor
            if self._use_fixed_spawn_draws
            else uniform(0.3, 0.7)
        )
        for u in [1, -1]:
            for v in [1, -1]:
                metric = np.where(z < leg_height, u * x + v * y, -np.inf)
                i = np.argmax(metric)
                p = co[i]
                n = normal[i]
                q = co[i] + self.leg_side * np.array(
                    [n[0], n[1] * self.leg_y_scale, n[2]]
                )
                r = np.array([q[0], q[1], 0])
                leg = new_line(2)
                write_co(leg, np.stack([p, q, r]))
                subsurf(leg, self.leg_subsurf_level)
                surface.add_geomod(
                    leg,
                    geo_radius,
                    apply=True,
                    input_args=[self.leg_radius, 32],
                    input_kwargs={"to_align_tilt": False},
                )
                butil.modify_mesh(
                    leg, "BEVEL", width=self.leg_radius * bevel_factor
                )
                leg.location[-1] = self.leg_radius
                butil.apply_transform(leg, True)
                write_attribute(leg, 1, "leg", "FACE")
                legs.append(leg)
        return legs

    def add_base(self, obj):
        obj = deep_clone_obj(obj)
        cutter = new_cube()
        x, y, z_ = read_co(obj).T
        cutter.scale = 10, 10, np.min(z_) + self.leg_height
        butil.apply_transform(cutter, True)
        butil.modify_mesh(obj, "BOOLEAN", object=cutter, operation="INTERSECT")
        butil.delete(cutter)
        with butil.ViewportMode(obj, "EDIT"):
            bm = bmesh.from_edit_mesh(obj.data)
            geom = [f for f in bm.faces if len(f.verts) > 10]
            bmesh.ops.delete(bm, geom=geom, context="FACES_KEEP_BOUNDARY")
            bmesh.update_edit_mesh(obj.data)
            bpy.ops.mesh.select_mode(type="EDGE")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.region_to_loop()
            bpy.ops.mesh.select_all(action="INVERT")
            bpy.ops.mesh.delete(type="EDGE")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.extrude_edges_move(
                TRANSFORM_OT_translate={"value": (0, 0, -self.depth)}
            )
        x, y, z = read_co(obj).T
        z = np.clip(z, 0, None)
        write_co(obj, np.stack([x, y, z], -1))
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.normals_make_consistent(inside=False)
        subsurf(obj, 2)
        butil.modify_mesh(obj, "SOLIDIFY", thickness=self.thickness)
        return obj

    def make_box_contour(self, t, i):
        return [
            (t + self.disp_x[0] * i, t + self.disp_y * i),
            (self.width - t - self.disp_x[1] * i, t + self.disp_y * i),
            (self.width - t - self.disp_x[1] * i, self.size - t - self.disp_y * i),
            (t + self.disp_x[0] * i, self.size - t - self.disp_y * i),
        ]

    def make_corner_contour(self, t, i):
        return [
            (t + self.disp_y * i, t + self.disp_y * i),
            (self.width - t - self.disp_x[1] * i, t + self.disp_y * i),
            (
                self.width - t - self.disp_x[1] * i,
                self.size - (t + self.disp_y * i) / np.sqrt(2),
            ),
            (
                self.size - (t + self.disp_y * i) / np.sqrt(2),
                self.width - t - self.disp_x[0] * i,
            ),
            (t + self.disp_y * i, self.width - t - self.disp_x[0] * i),
        ]

    def make_base(self):
        contour = self.contour_fn(0, 1)
        obj = new_cylinder(vertices=len(contour))
        co = np.concatenate(
            [np.array([[x, y, 0], [x, y, self.depth]]) for x, y in contour]
        )
        write_co(obj, co)
        return obj

    def make_bowl(self):
        if self.has_curve:
            lower = self.contour_fn(0, 1)
            upper = self.contour_fn(0, -1)
        else:
            lower = self.contour_fn(0, 1)
            upper = self.contour_fn(0, -1)
        obj = new_cylinder(vertices=len(lower))
        co = np.concatenate(
            [
                np.array([[x, y, 0], [z, w, self.depth * 2]])
                for (x, y), (z, w) in zip(lower[::-1], upper[::-1])
            ]
        )
        write_co(obj, co)
        subsurf(obj, self.alcove_levels, True)
        levels = self.levels - self.alcove_levels - self.side_levels
        subsurf(obj, levels)
        return obj

    def make_cutter(self):
        if self.has_curve:
            lower = self.contour_fn(self.thickness, 1)
            upper = self.contour_fn(self.thickness, -1)
        else:
            lower = self.contour_fn(self.thickness, 1)
            upper = self.contour_fn(self.thickness, -1)
        obj = new_cylinder(vertices=len(lower))
        co = np.concatenate(
            [
                np.array(
                    [[x, y, self.thickness], [z, w, self.depth * 2 - self.thickness]]
                )
                for (x, y), (z, w) in zip(lower[::-1], upper[::-1])
            ]
        )
        write_co(obj, co)
        subsurf(obj, self.alcove_levels, True)
        levels = self.levels - self.alcove_levels
        subsurf(obj, levels)
        return obj

    def find_hole(self, obj, x=None, y=None):
        if x is None:
            x = self.width / 2
        if y is None:
            y = self.size / 2
        up_facing = read_normal(obj)[:, -1] > 0
        center = read_center(obj)
        i = np.argmin(np.abs(center[:, :2] - np.array([[x, y]])).sum(1) - up_facing)
        return center[i]

    def add_hole(self, obj):
        match self.bathtub_type:
            case "alcove":
                location = self.find_hole(obj)
            case "freestanding":
                hole_x = (
                    self.hole_x_ratio * self.width
                    if self._use_fixed_spawn_draws
                    else uniform(0.35, 0.4) * self.width
                )
                location = self.find_hole(obj, hole_x)
            case _:
                location = self.find_hole(obj, self.size / 2, self.size / 2)
        if self.is_hole_centered:
            location = self.find_hole(obj)
        obj = new_cylinder()
        obj.scale = self.hole_radius, self.hole_radius, 0.005
        obj.location = location
        butil.apply_transform(obj, True)
        write_attribute(obj, 1, "hole", "FACE")
        return obj

    def finalize_assets(self, assets):
        self.surface.apply(assets, clear=True)
        if self.has_legs and not self.has_base:
            self.leg_surface.apply(assets, "leg", metal_color="bw+natural")
        self.hole_surface.apply(assets, "hole", metal_color="bw+natural")

        if self.scratch:
            self.scratch.apply(assets)
        if self.edge_wear:
            self.edge_wear.apply(assets)
