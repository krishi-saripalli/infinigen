# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


from __future__ import annotations

from typing import Annotated, ClassVar

import bpy
import numpy as np
from numpy.random import uniform
from pydantic import Field

from infinigen.assets.objects.monocot.growth import MonocotGrowthFactory
from infinigen.assets.utils.decorate import (
    remove_vertices,
    write_attribute,
    write_material_index,
)
from infinigen.assets.utils.draw import bezier_curve, leaf, spin
from infinigen.assets.utils.mesh import polygon_angles
from infinigen.assets.utils.misc import assign_material
from infinigen.assets.utils.object import join_objects
from infinigen.core import surface
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.placement.detail import remesh_with_attrs
from infinigen.core.placement.factory import AssetFactory, make_asset_collection
from infinigen.core.placement.parameters import (
    AssetParameters,
    ParameterizedAssetFactory,
)
from infinigen.core.surface import shaderfunc_to_material
from infinigen.core.tagging import tag_object
from infinigen.core.util import blender as butil
from infinigen.core.util.color import hsv2rgba
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform


class GrassesMonocotParameters(AssetParameters):
    angle: Annotated[float, Field(ge=0.523599, le=1.047198, json_schema_extra={"editable": True})]
    z_drag: Annotated[float, Field(ge=0.0, le=0.2, json_schema_extra={"editable": False})]
    min_y_angle: Annotated[
        float, Field(ge=1.099557, le=1.413717, json_schema_extra={"editable": True})
    ]
    max_y_angle: Annotated[
        float, Field(ge=1.413717, le=1.570796, json_schema_extra={"editable": False})
    ]
    count: Annotated[float, Field(ge=16.0, le=64.0, json_schema_extra={"editable": True})]
    base_hue_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": False})
    ] = 0.3
    # NOTE: probabilistic leaf spawn; no geometry change when leaf draw fails.
    leaf_prob: Annotated[float, Field(ge=0.8, le=0.9, json_schema_extra={"editable": False})]
    cut_prob_draw: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            json_schema_extra={"editable": False, "kind": "draw_bool", "threshold": 0.4},
        ),
    ] = 0.0
    trim_leaf: Annotated[
        bool, Field(json_schema_extra={"editable": False, "kind": "bool"})
    ] = False


class GrassesMonocotFactory(ParameterizedAssetFactory, MonocotGrowthFactory):
    parameters_model: ClassVar[type[AssetParameters]] = GrassesMonocotParameters

    def __init__(self, factory_seed, coarse=False):
        AssetFactory.__init__(self, factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_material(self, seed: int) -> None:
        # NOTE: base_hue is sampled on self in apply_parameters; excluded from quartet sampling (material-only, not exported geometry).
        with FixedSeed(seed):
            base_hue_draw = uniform(0, 1)
            if base_hue_draw < 0.6:
                base_hue = uniform(0.08, 0.12)
            else:
                base_hue = uniform(0.2, 0.25)
            bright_color = hsv2rgba(base_hue, uniform(0.6, 0.8), log_uniform(0.05, 0.1))
            dark_color = hsv2rgba(
                (base_hue + uniform(-0.03, 0.03)) % 1,
                uniform(0.8, 1.0),
                log_uniform(0.05, 0.2),
            )
        self.base_hue = base_hue
        self.material = shaderfunc_to_material(
            self.shader_monocot, dark_color, bright_color, self.use_distance
        )

    def _sample_init_parameters(self, seed: int) -> GrassesMonocotParameters:
        base_hue_draw = uniform(0, 1)
        leaf_prob = uniform(0.8, 0.9)
        self._sample_material(seed)
        return GrassesMonocotParameters(
            seed=seed,
            angle=uniform(np.pi / 6, np.pi / 3),
            z_drag=uniform(0.0, 0.2),
            min_y_angle=uniform(np.pi * 0.35, np.pi * 0.45),
            max_y_angle=uniform(np.pi * 0.45, np.pi * 0.5),
            count=log_uniform(16, 64),
            base_hue_draw=base_hue_draw,
            leaf_prob=leaf_prob,
        )

    def _sample_spawn_parameters(
        self, params: GrassesMonocotParameters, seed: int, i: int
    ) -> GrassesMonocotParameters:
        return params.model_copy(
            update={"cut_prob_draw": uniform(), "trim_leaf": params.trim_leaf}
        )

    def apply_parameters(
        self, params: GrassesMonocotParameters, *, spawn_scope: bool = True
    ) -> None:
        self._sample_material(params.seed)
        # NOTE: stem_offset and z_scale do not elicit a clear visual change in exported geometry; excluded from quartet sampling.
        with FixedSeed(params.seed):
            if isinstance(params, MaizeMonocotParameters):
                self.stem_offset = params.stem_offset
            else:
                self.stem_offset = uniform(1.5, 2.0)
            self.z_scale = uniform(1.0, 1.2)
        self.angle = params.angle
        self.z_drag = params.z_drag
        if isinstance(params, MaizeMonocotParameters):
            # NOTE: min_y_angle does not elicit a clear visual change in exported geometry; excluded from quartet sampling.
            with FixedSeed(params.seed):
                self.min_y_angle = uniform(np.pi * 0.35, np.pi * 0.45)
        else:
            self.min_y_angle = params.min_y_angle
        self.max_y_angle = params.max_y_angle
        self.count = int(params.count)
        self.scale_curve = [(0, 1.0), (1, 0.2)]
        self.bend_angle = np.pi / 2
        self.leaf_prob = params.leaf_prob
        self.cut_prob_draw = params.cut_prob_draw
        self.trim_leaf = params.cut_prob_draw < 0.4
        self.leaf_range = (0, 1)
        self.radius = 0.01
        self.perturb = 0.05
        self.twist_angle = np.pi / 6
        self.align_factor = 0
        self.align_direction = (1, 0, 0)
        self._cache_decor_state(params.seed)
        self._use_fixed_spawn_draws = spawn_scope

    @staticmethod
    def build_base_hue():
        if uniform(0, 1) < 0.6:
            return uniform(0.08, 0.12)
        else:
            return uniform(0.2, 0.25)

    def build_leaf(self, face_size):
        x_anchors = np.array([0, uniform(0.1, 0.2), uniform(0.5, 0.7), 1.0])
        y_anchors = np.array([0, uniform(0.02, 0.03), uniform(0.02, 0.03), 0])
        obj = leaf(x_anchors, y_anchors, face_size=face_size)

        if self.trim_leaf if self._use_fixed_spawn_draws else uniform() < 0.4:
            x_cutoff = uniform(0.5, 1.0)
            angle = uniform(-np.pi / 3, np.pi / 3)
            remove_vertices(
                obj,
                lambda x, y, z: (x - x_cutoff) * np.cos(angle) + y * np.sin(angle) > 0,
            )
        self.decorate_leaf(obj)
        tag_object(obj, "grasses")
        return obj

    @property
    def is_grass(self):
        return True


class WheatEarMonocotParameters(AssetParameters):
    angle: Annotated[float, Field(ge=0.523599, le=0.785398, json_schema_extra={"editable": False})]
    min_y_angle: Annotated[
        float, Field(ge=0.785398, le=1.047198, json_schema_extra={"editable": False})
    ]
    z_scale: Annotated[float, Field(ge=1.0, le=1.2, json_schema_extra={"editable": False})]


class WheatEarMonocotFactory(ParameterizedAssetFactory, MonocotGrowthFactory):
    parameters_model: ClassVar[type[AssetParameters]] = WheatEarMonocotParameters

    def __init__(self, factory_seed, coarse=False):
        AssetFactory.__init__(self, factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_material(self, seed: int) -> None:
        # NOTE: base_hue is sampled on self in apply_parameters; excluded from quartet sampling (material-only, not exported geometry).
        with FixedSeed(seed):
            base_hue = uniform(0.12, 0.28)
            bright_color = hsv2rgba(base_hue, uniform(0.6, 0.8), log_uniform(0.05, 0.1))
            dark_color = hsv2rgba(
                (base_hue + uniform(-0.03, 0.03)) % 1,
                uniform(0.8, 1.0),
                log_uniform(0.05, 0.2),
            )
        self.base_hue = base_hue
        self.material = shaderfunc_to_material(
            self.shader_monocot, dark_color, bright_color, self.use_distance
        )

    def _sample_init_parameters(self, seed: int) -> WheatEarMonocotParameters:
        z_scale = uniform(1.0, 1.2)
        self._sample_material(seed)
        return WheatEarMonocotParameters(
            seed=seed,
            angle=uniform(np.pi / 6, np.pi / 4),
            min_y_angle=uniform(np.pi / 4, np.pi / 3),
            z_scale=z_scale,
        )

    def apply_parameters(
        self, params: WheatEarMonocotParameters, *, spawn_scope: bool = True
    ) -> None:
        self._sample_material(params.seed)
        # NOTE: stem_offset, leaf_prob, count ranges too narrow to survive cube normalization; sampled on self, excluded from quartet sampling.
        with FixedSeed(params.seed):
            self.stem_offset = uniform(0.4, 0.5)
            self.leaf_prob = uniform(0.9, 1)
            self.count = int(log_uniform(96, 128))
        # NOTE: angle, min_y_angle, and z_scale have intermittent effect across ear leaf instances; excluded from quartet sampling.
        self.angle = params.angle
        self.min_y_angle = params.min_y_angle
        self.max_y_angle = np.pi / 2
        self.bend_angle = np.pi
        self.z_scale = params.z_scale
        self.leaf_range = (0, 1)
        self.radius = 0.01
        self.perturb = 0.05
        self.z_drag = 0.0
        self.twist_angle = np.pi / 6
        self.scale_curve = [(0, 1), (1, 1)]
        self.align_factor = 0
        self.align_direction = (1, 0, 0)
        self._cache_decor_state(params.seed)
        self._use_fixed_spawn_draws = spawn_scope

    @staticmethod
    def build_base_hue():
        return uniform(0.12, 0.28)

    def build_leaf(self, face_size):
        x_anchors = np.array([0, 0.05, 0.1])
        y_anchors = np.array([0, uniform(0.01, 0.015), 0])
        curves = []
        for angle in polygon_angles(np.random.randint(4, 6)):
            anchors = [x_anchors, np.cos(angle) * y_anchors, np.sin(angle) * y_anchors]
            curves.append(bezier_curve(anchors))
        obj = butil.join_objects(curves)
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.convex_hull()
        remesh_with_attrs(obj, face_size / 2)
        tag_object(obj, "wheat_ear")
        return obj


class WheatMonocotParameters(GrassesMonocotParameters):
    ear_bend_angle: Annotated[
        float, Field(ge=0.0, le=np.pi, json_schema_extra={"editable": False})
    ] = 0.0


class WheatMonocotFactory(GrassesMonocotFactory):
    parameters_model: ClassVar[type[AssetParameters]] = WheatMonocotParameters

    def __init__(self, factory_seed, coarse=False):
        AssetFactory.__init__(self, factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> WheatMonocotParameters:
        params = super()._sample_init_parameters(seed)
        self.ear_factory = WheatEarMonocotFactory(seed, self.coarse)
        return WheatMonocotParameters(**params.model_dump())

    def _sample_spawn_parameters(
        self, params: WheatMonocotParameters, seed: int, i: int
    ) -> WheatMonocotParameters:
        params = super()._sample_spawn_parameters(params, seed, i)
        return params.model_copy(
            update={"ear_bend_angle": uniform(0, np.pi)}
        )

    def apply_parameters(
        self, params: WheatMonocotParameters, *, spawn_scope: bool = True
    ) -> None:
        super().apply_parameters(params, spawn_scope=spawn_scope)
        self.scale_curve = [(0, 1.0), (1, 0.6)]
        self.leaf_range = (0.1, 0.7)
        if not hasattr(self, "ear_factory"):
            self.ear_factory = WheatEarMonocotFactory(params.seed, self.coarse)
        if spawn_scope:
            self.ear_factory.bend_angle = params.ear_bend_angle

    @staticmethod
    def build_base_hue():
        return uniform(0.08, 0.12)

    def create_asset(self, **params):
        obj = super().create_raw(**params)
        ear = self.ear_factory.create_asset(**params)
        bend_angle = (
            self.ear_bend_angle
            if self._use_fixed_spawn_draws
            else uniform(0, self.ear_factory.bend_angle)
        )
        butil.modify_mesh(
            ear,
            "SIMPLE_DEFORM",
            deform_method="BEND",
            angle=bend_angle,
        )
        ear.location[-1] = self.stem_offset - 0.02
        obj = join_objects([obj, ear])
        self.decorate_monocot(obj)
        tag_object(obj, "wheat")
        return obj


class MaizeMonocotParameters(GrassesMonocotParameters):
    stem_offset: Annotated[float, Field(ge=2.0, le=2.5, json_schema_extra={"editable": False})]
    trim_leaf: Annotated[
        bool, Field(json_schema_extra={"editable": False, "kind": "bool"})
    ] = False


MaizeMonocotParameters.model_fields.pop("min_y_angle")
MaizeMonocotParameters.model_fields.pop("z_drag")
MaizeMonocotParameters.model_rebuild(force=True)


class MaizeMonocotFactory(GrassesMonocotFactory):
    parameters_model: ClassVar[type[AssetParameters]] = MaizeMonocotParameters

    def __init__(self, factory_seed, coarse=False):
        AssetFactory.__init__(self, factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> MaizeMonocotParameters:
        base_hue_draw = uniform(0, 1)
        leaf_prob = uniform(0.8, 0.9)
        self._sample_material(seed)
        return MaizeMonocotParameters(
            seed=seed,
            stem_offset=uniform(2.0, 2.5),
            angle=uniform(np.pi / 6, np.pi / 3),
            max_y_angle=uniform(np.pi * 0.45, np.pi * 0.5),
            count=log_uniform(16, 64),
            base_hue_draw=base_hue_draw,
            leaf_prob=leaf_prob,
        )

    def apply_parameters(
        self, params: MaizeMonocotParameters, *, spawn_scope: bool = True
    ) -> None:
        super().apply_parameters(params, spawn_scope=spawn_scope)
        with FixedSeed(params.seed):
            self.z_drag = uniform(0.0, 0.2)
        self.scale_curve = [(0, 1.0), (1, 0.6)]
        self.leaf_range = (0.1, 0.7)

    def build_leaf(self, face_size):
        x_anchors = np.array([0, uniform(0.1, 0.2), uniform(0.5, 0.7), 1.0])
        y_anchors = np.array([0, uniform(0.03, 0.06), uniform(0.03, 0.06), 0])
        obj = leaf(x_anchors, y_anchors, face_size=face_size)
        self.decorate_leaf(obj)
        tag_object(obj, "maize_leaf")
        return obj

    def build_husk(self):
        x_anchors = 0, uniform(0.04, 0.05), uniform(0.03, 0.03), 0
        z_anchors = 0, 0.01, uniform(0.24, 0.3), uniform(0.35, 0.4)
        anchors = x_anchors, 0, z_anchors
        husk = spin(anchors)
        texture = bpy.data.textures.new(name="husk", type="STUCCI")
        texture.noise_scale = 0.01
        butil.modify_mesh(husk, "DISPLACE", strength=0.02, texture=texture)
        husk.location[-1] = self.stem_offset - 0.02
        husk.rotation_euler[0] = uniform(0, np.pi * 0.2)
        tag_object(husk, "maize_husk")
        return husk

    def create_asset(self, **params):
        obj = super().create_raw(**params)
        husk = self.build_husk()
        obj = join_objects([obj, husk])
        self.decorate_monocot(obj)
        tag_object(obj, "maize")
        return obj


class ReedEarMonocotFactory(MonocotGrowthFactory):
    def __init__(self, factory_seed, coarse=False):
        super(ReedEarMonocotFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.stem_offset = uniform(0.3, 0.4)
            self.min_y_angle = uniform(np.pi / 4, np.pi / 3)
            self.max_y_angle = self.min_y_angle + np.pi / 12
            self.count = int(log_uniform(48, 96))
            self.radius = 0.002

    def build_leaf(self, face_size):
        x_anchors = np.array([0, uniform(0.02, 0.03), 0.05])
        y_anchors = np.array([0, uniform(0.005, 0.01), 0])
        obj = leaf(x_anchors, y_anchors, face_size=face_size)
        return obj

    def create_raw(self, **params):
        obj = super(ReedEarMonocotFactory, self).create_raw(**params)
        write_attribute(obj, 1, "ear", "FACE")
        tag_object(obj, "reed_ear")
        return obj


class ReedBranchMonocotFactory(MonocotGrowthFactory):
    max_branches = 6

    def __init__(self, factory_seed, coarse=False):
        super(ReedBranchMonocotFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.stem_offset = uniform(0.6, 0.8)
            self.ear_factory = ReedEarMonocotFactory(self.factory_seed)
            self.scale_curve = (0, 1), (0.5, 0.6), (1, 0.1)
            self.min_y_angle = uniform(-np.pi / 10, -np.pi / 8)
            self.max_y_angle = uniform(-np.pi / 6, -np.pi / 8)
            self.angle = 0
            self.radius = 0.005

    def make_collection(self, face_size):
        return make_asset_collection(
            self.ear_factory.create_raw, 2, "leaves", verbose=False, face_size=face_size
        )


class ReedMonocotFactory(GrassesMonocotFactory):
    def __init__(self, factory_seed, coarse=False):
        super(ReedMonocotFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.stem_offset = uniform(3.0, 4.0)
            self.scale_curve = [(0, 1.2), (1, 0.8)]
            self.branch_factory = ReedBranchMonocotFactory(factory_seed, coarse)
            self.branch_material = shaderfunc_to_material(self.shader_ear)

    @staticmethod
    def build_base_hue():
        return uniform(0.08, 0.12)

    def create_asset(self, **params):
        obj = super().create_raw(**params)
        branch = self.branch_factory.create_asset(**params)
        self.branch_factory.decorate_monocot(branch)
        branch.location[-1] = self.stem_offset - 0.02
        obj = join_objects([obj, branch])
        butil.modify_mesh(obj, "WELD", merge_threshold=1e-3)
        self.decorate_monocot(obj)

        assign_material(obj, [self.material, self.branch_material])
        write_material_index(
            obj, surface.read_attr_data(obj, "ear", "FACE").astype(int)[:, 0]
        )
        tag_object(obj, "reed")
        return obj

    @staticmethod
    def shader_ear(nw: NodeWrangler):
        color = hsv2rgba(uniform(0.06, 0.1), uniform(0.2, 0.5), log_uniform(0.2, 0.5))
        specular = uniform(0.0, 0.2)
        clearcoat = 0 if uniform(0, 1) < 0.8 else uniform(0.2, 0.5)
        noise_texture = nw.new_node(Nodes.NoiseTexture, input_kwargs={"Scale": 50})
        roughness = nw.build_float_curve(noise_texture, [(0, 0.5), (1, 0.8)])
        bsdf = nw.new_node(
            Nodes.PrincipledBSDF,
            input_kwargs={
                "Base Color": color,
                "Roughness": roughness,
                "Specular IOR Level": specular,
                "Coat Weight": clearcoat,
                "Subsurface Weight": 0.01,
                "Subsurface Radius": (0.01, 0.01, 0.01),
            },
        )
        return bsdf
