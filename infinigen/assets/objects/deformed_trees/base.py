# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Lingjie Mei


from __future__ import annotations

import colorsys
from typing import Annotated, ClassVar

from numpy.random import uniform
from pydantic import Field

from infinigen.assets.composition import material_assignments
from infinigen.assets.objects.trees.generate import GenericTreeFactory, random_species
from infinigen.core import surface
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import (
    AssetParameters,
    ParameterizedAssetFactory,
)
from infinigen.core.surface import NoApply
from infinigen.core.util.color import hsv2rgba
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform, weighted_sample


class BaseDeformedTreeParameters(AssetParameters):
    base_hue: Annotated[
        float, Field(ge=0.02, le=0.08, json_schema_extra={"editable": True})
    ]
    skinning_scale: Annotated[
        float, Field(ge=0.15, le=0.25, json_schema_extra={"editable": True})
    ]
    ring_wave_scale: Annotated[
        float, Field(ge=10.0, le=20.0, json_schema_extra={"editable": True})
    ]
    ring_distortion: Annotated[
        float, Field(ge=4.0, le=10.0, json_schema_extra={"editable": True})
    ]


class BaseDeformedTreeFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = BaseDeformedTreeParameters

    def __init__(self, factory_seed, coarse=False):
        AssetFactory.__init__(self, factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> BaseDeformedTreeParameters:
        self._trunk_surface = weighted_sample(material_assignments.bark)
        return BaseDeformedTreeParameters(
            seed=seed,
            base_hue=uniform(0.02, 0.08),
            skinning_scale=uniform(0.15, 0.25),
            ring_wave_scale=uniform(10, 20),
            ring_distortion=uniform(4, 10),
        )

    def apply_parameters(
        self, params: BaseDeformedTreeParameters, *, spawn_scope: bool = True
    ) -> None:
        if not hasattr(self, "_trunk_surface"):
            self._trunk_surface = weighted_sample(material_assignments.bark)
        self.base_hue = params.base_hue
        self.skinning_scale = params.skinning_scale
        self.ring_wave_scale = params.ring_wave_scale
        self.ring_distortion = params.ring_distortion
        self.trunk_surface = self._trunk_surface
        self._use_fixed_spawn_draws = spawn_scope

    def _run_post_init(self) -> None:
        with FixedSeed(self.factory_seed):
            (tree_params, _, _), _ = random_species()
            tree_params.skinning.update({"Scaling": self.skinning_scale})
            self.base_factory = GenericTreeFactory(
                self.factory_seed, tree_params, None, NoApply, self.coarse
            )
            self.material = surface.shaderfunc_to_material(
                self.shader_rings, self.base_hue
            )

    def build_tree(self, i, distance, **kwargs):
        return self.base_factory.spawn_asset(i=i, distance=distance)

    @staticmethod
    def geo_xyz(nw: NodeWrangler):
        geometry = nw.new_node(
            Nodes.GroupInput, expose_input=[("NodeSocketGeometry", "Geometry", None)]
        )
        for name, component in zip(
            "xyz", nw.separate(nw.new_node(Nodes.InputPosition))
        ):
            geometry = nw.new_node(
                Nodes.StoreNamedAttribute,
                input_kwargs={"Geometry": geometry, "Name": name, "Value": component},
            )
        nw.new_node(Nodes.GroupOutput, input_kwargs={"Geometry": geometry})

    def shader_rings(self, nw: NodeWrangler, base_hue):
        position = nw.combine(
            *map(
                lambda n: nw.new_node(Nodes.Attribute, attrs={"attribute_name": n}),
                "xyz",
            )
        )
        ratio = nw.new_node(
            Nodes.WaveTexture,
            [position],
            input_kwargs={
                "Scale": self.ring_wave_scale,
                "Distortion": self.ring_distortion,
            },
            attrs={"wave_type": "RINGS", "rings_direction": "Z", "wave_profile": "SAW"},
        )
        bright_color = hsv2rgba(base_hue, uniform(0.4, 0.8), log_uniform(0.2, 0.8))
        dark_color = (
            *colorsys.hsv_to_rgb(
                (base_hue + uniform(-0.02, 0.02)) % 1,
                uniform(0.4, 0.8),
                log_uniform(0.02, 0.05),
            ),
            1.0,
        )
        color = nw.new_node(Nodes.MixRGB, [ratio, dark_color, bright_color])
        principled_bsdf = nw.new_node(
            Nodes.PrincipledBSDF, input_kwargs={"Base Color": color}
        )
        return principled_bsdf

    def create_asset(self, face_size, **params):
        raise NotImplementedError
