# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei

from __future__ import annotations

from typing import Annotated, Any, ClassVar, Type

import colorsys

import bpy
import numpy as np
from numpy.random import uniform
from pydantic import Field

from infinigen.assets.utils.decorate import subsurface2face_size
from infinigen.assets.utils.draw import shape_by_angles
from infinigen.assets.utils.misc import assign_material
from infinigen.assets.utils.object import join_objects
from infinigen.core import surface
from infinigen.core.nodes.node_utils import build_color_ramp
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import AssetParameters, ParameterizedAssetFactory
from infinigen.core.tagging import tag_object
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform

from .base import BaseMolluskFactory
from .shell import (
    ClamBaseFactory,
    MusselBaseFactory,
    ScallopBaseFactory,
    ShellBaseFactory,
)
from .snail import (
    AugerBaseFactory,
    ConchBaseFactory,
    NautilusBaseFactory,
    SnailBaseFactory,
    VoluteBaseFactory,
)


class MolluskParameters(AssetParameters):
    pass


class MusselParameters(MolluskParameters):
    accent_hue: Annotated[
        float, Field(ge=0.05, le=0.12, json_schema_extra={"editable": False})
    ]
    z_scale: Annotated[float, Field(ge=2.0, le=10.0, json_schema_extra={"editable": False})] = (
        5.0
    )


class MolluskFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = MolluskParameters
    _factory_method_class: ClassVar[Type | None] = None

    def __init__(self, factory_seed, coarse=False, factory_method=None):
        super(MolluskFactory, self).__init__(factory_seed, coarse)
        self._init_factory_method = factory_method
        self.init_legacy_parameters()

    def _resolve_factory_method(self, factory_method=None):
        if factory_method is not None:
            return factory_method
        if self._factory_method_class is not None:
            return self._factory_method_class
        factory_methods = [SnailBaseFactory, ShellBaseFactory]
        weights = np.array([1] * len(factory_methods))
        return np.random.choice(factory_methods, p=weights / weights.sum())

    def _sample_base_hue(self, accent_hue: float | None = None) -> float:
        if uniform(0, 1) < 0.4:
            return uniform(0, 0.2)
        if accent_hue is not None:
            return accent_hue
        return uniform(0.05, 0.12)

    def _build_factory(self, factory_method, seed: int) -> BaseMolluskFactory:
        return factory_method(seed, self.coarse)

    def _sample_shader_state(self, seed: int, accent_hue: float | None = None) -> None:
        # NOTE: base_hue and texture_type_draw are sampled on self in apply_parameters; excluded from quartet sampling (material-only, not exported geometry).
        with FixedSeed(seed):
            factory_method = self._resolve_factory_method(self._init_factory_method)
            factory = self._build_factory(factory_method, seed)
            base_hue = self._sample_base_hue(accent_hue)
            texture_type_draw = uniform()
        self.factory = factory
        self.base_hue = base_hue
        self.material = surface.shaderfunc_to_material(
            self.shader_mollusk,
            base_hue,
            factory.ratio,
            factory.x_scale,
            factory.z_scale,
            factory.distortion,
        )
        self._texture_type_draw = texture_type_draw

    def _sample_init_parameters(self, seed: int) -> MolluskParameters:
        self._sample_shader_state(seed)
        return MolluskParameters(seed=seed)

    def apply_parameters(
        self, params: MolluskParameters, *, spawn_scope: bool = True
    ) -> None:
        self._sample_shader_state(params.seed)
        with FixedSeed(params.seed):
            self._noise_scale = log_uniform(0.1, 0.2)
        self._use_fixed_spawn_draws = spawn_scope

    def create_asset(self, face_size=0.01, **params):
        obj = self.factory.create_asset(**params)
        self.decorate_mollusk(obj, face_size)
        return obj

    def decorate_mollusk(self, obj, face_size):
        subsurface2face_size(obj, face_size)
        butil.modify_mesh(obj, "SOLIDIFY", True, thickness=0.005)
        texture_types = ["STUCCI", "MARBLE"]
        t = (
            texture_types[
                min(
                    int(self._texture_type_draw * len(texture_types)),
                    len(texture_types) - 1,
                )
            ]
            if self._use_fixed_spawn_draws
            else np.random.choice(texture_types)
        )
        texture = bpy.data.textures.new(name="mollusk", type=t)
        noise_scale = (
            self._noise_scale if self._use_fixed_spawn_draws else log_uniform(0.1, 0.2)
        )
        texture.noise_scale = noise_scale
        butil.modify_mesh(
            obj,
            "DISPLACE",
            strength=self.factory.noise_strength,
            mid_level=0,
            texture=texture,
        )
        assign_material(obj, self.material)
        tag_object(obj, "mollusk")
        return obj

    @staticmethod
    def build_base_hue():
        if uniform(0, 1) < 0.4:
            return uniform(0, 0.2)
        else:
            return uniform(0.05, 0.12)

    @staticmethod
    def shader_mollusk(
        nw: NodeWrangler, base_hue, ratio=0, x_scale=2, z_scale=1, distortion=5
    ):
        roughness = uniform(0.2, 0.8)
        specular = 0.3
        value_scale = log_uniform(1, 20)
        saturation_scale = log_uniform(0.4, 1)

        def dark_color():
            return *colorsys.hsv_to_rgb(
                base_hue + uniform(-0.06, 0.06),
                uniform(0.6, 1.0) * saturation_scale,
                0.005 * value_scale**1.5,
            ), 1

        def light_color():
            return *colorsys.hsv_to_rgb(
                base_hue + uniform(-0.06, 0.06),
                uniform(0.6, 1.0) * saturation_scale,
                0.05 * value_scale,
            ), 1

        def color_fn(dark_prob):
            return dark_color() if uniform(0, 1) < dark_prob else light_color()

        vector = nw.new_node(
            Nodes.Attribute, attrs={"attribute_name": "vector"}
        ).outputs["Vector"]
        n = np.random.randint(3, 5)
        texture_0 = nw.new_node(
            Nodes.WaveTexture,
            input_kwargs={"Vector": vector, "Distortion": distortion, "Scale": x_scale},
            attrs={"wave_profile": "SAW", "bands_direction": "X"},
        )
        cr_0 = build_color_ramp(
            nw, texture_0, np.sort(uniform(0, 1, n)), [color_fn(0.4) for _ in range(n)]
        )
        texture_1 = nw.new_node(
            Nodes.WaveTexture,
            input_kwargs={"Vector": vector, "Distortion": distortion, "Scale": z_scale},
            attrs={"wave_profile": "SAW", "bands_direction": "Z"},
        )
        cr_1 = build_color_ramp(
            nw, texture_1, np.sort(uniform(0, 1, n)), [color_fn(0.4) for _ in range(n)]
        )
        principled_bsdf = nw.new_node(
            Nodes.PrincipledBSDF,
            input_kwargs={
                "Base Color": nw.new_node(Nodes.MixRGB, [ratio, cr_0, cr_1]),
                "Specular IOR Level": specular,
                "Roughness": roughness,
            },
        )
        return principled_bsdf


def _bind_mussel_shell_factory(
    factory: MusselBaseFactory,
    *,
    shell_open_angle: float,
    shell_tilt: float,
    profile_mid_scale: float,
    profile_tip_scale: float,
    fixed: bool,
) -> None:
    if not fixed:
        return

    def mussel_make():
        obj = factory.build_ellipse(softness=0.5)
        obj.scale = 1, 3, 1
        butil.apply_transform(obj)
        angles = [-0.5, -uniform(0.1, 0.15), uniform(0.0, 0.25), 0.5]
        scale_values = [0, profile_mid_scale, 1, profile_tip_scale]
        shape_by_angles(obj, np.array(angles) * np.pi, scale_values)
        tag_object(obj, "mussel")
        return obj

    def create_asset(**kw):
        upper = mussel_make()
        dim = np.sqrt(upper.dimensions[0] * upper.dimensions[1] + 0.01)
        upper.scale = [1 / dim] * 3
        upper.location[-1] += 0.005
        butil.apply_transform(upper, loc=True)
        lower = butil.deep_clone_obj(upper)
        lower.scale[-1] = -1
        butil.apply_transform(lower)
        lower.rotation_euler[1] = -shell_tilt
        upper.rotation_euler[1] = -shell_tilt - shell_open_angle
        return join_objects([lower, upper])

    factory.mussel_make = mussel_make
    factory.create_asset = create_asset


class MusselFactory(MolluskFactory):
    parameters_model: ClassVar[type[AssetParameters]] = MusselParameters
    _factory_method_class = MusselBaseFactory

    def _sample_init_parameters(self, seed: int) -> MusselParameters:
        with FixedSeed(seed):
            factory_method = self._resolve_factory_method(self._init_factory_method)
            accent_hue = uniform(0.05, 0.12)
            factory = self._build_factory(factory_method, seed)
            factory.z_scale = log_uniform(2, 10)
            base_hue = self._sample_base_hue(accent_hue)
            texture_type_draw = uniform()
        self.factory = factory
        self.base_hue = base_hue
        self.material = surface.shaderfunc_to_material(
            self.shader_mollusk,
            base_hue,
            factory.ratio,
            factory.x_scale,
            factory.z_scale,
            factory.distortion,
        )
        self._texture_type_draw = texture_type_draw
        return MusselParameters(
            seed=seed,
            accent_hue=accent_hue,
            z_scale=factory.z_scale,
        )

    def apply_parameters(
        self, params: MusselParameters, *, spawn_scope: bool = True
    ) -> None:
        with FixedSeed(params.seed):
            factory_method = self._resolve_factory_method(self._init_factory_method)
            self.factory = self._build_factory(factory_method, params.seed)
            base_hue = self._sample_base_hue(params.accent_hue)
            self._texture_type_draw = uniform()
        self.factory.z_scale = params.z_scale
        self.base_hue = base_hue
        self.material = surface.shaderfunc_to_material(
            self.shader_mollusk,
            base_hue,
            self.factory.ratio,
            self.factory.x_scale,
            self.factory.z_scale,
            self.factory.distortion,
        )
        # NOTE: noise_scale and mussel shell profile params resampled in spawn path overwrote edits; sampled on self from seed, excluded from quartet sampling.
        with FixedSeed(params.seed):
            self._noise_scale = log_uniform(0.1, 0.2)
            self._shell_open_angle = uniform(np.pi / 6, np.pi / 3)
            self._shell_tilt = uniform(0, np.pi / 4)
            self._mussel_profile_mid_scale = uniform(0.6, 0.8)
            self._mussel_profile_tip_scale = uniform(0.6, 0.8)
        self._use_fixed_spawn_draws = spawn_scope

    def create_asset(self, face_size=0.01, **params):
        if self._use_fixed_spawn_draws:
            _bind_mussel_shell_factory(
                self.factory,
                shell_open_angle=self._shell_open_angle,
                shell_tilt=self._shell_tilt,
                profile_mid_scale=self._mussel_profile_mid_scale,
                profile_tip_scale=self._mussel_profile_tip_scale,
                fixed=True,
            )
        return super().create_asset(face_size=face_size, **params)


class ScallopFactory(MolluskFactory):
    _factory_method_class = ScallopBaseFactory


class ClamFactory(MolluskFactory):
    _factory_method_class = ClamBaseFactory


class ConchFactory(MolluskFactory):
    _factory_method_class = ConchBaseFactory


class AugerFactory(MolluskFactory):
    _factory_method_class = AugerBaseFactory


class VoluteFactory(MolluskFactory):
    _factory_method_class = VoluteBaseFactory


class NautilusFactory(MolluskFactory):
    parameters_model: ClassVar[type[AssetParameters]] = MolluskParameters
    _factory_method_class = NautilusBaseFactory
