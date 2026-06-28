# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.
# Authors: Lingjie Mei


from __future__ import annotations

from typing import Annotated, ClassVar

from numpy.random import uniform
from pydantic import Field

from infinigen.assets.objects.deformed_trees.fallen import FallenTreeFactory
from infinigen.assets.objects.deformed_trees.hollow import HollowTreeFactory
from infinigen.assets.objects.deformed_trees.rotten import RottenTreeFactory
from infinigen.assets.objects.deformed_trees.truncated import TruncatedTreeFactory
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import (
    AssetParameters,
    ParameterizedAssetFactory,
)
from infinigen.core.util.math import FixedSeed


class DeformedTreeParameters(AssetParameters):
    maker_draw: Annotated[
        str,
        Field(
            json_schema_extra={
                "editable": True,
                "kind": "enum",
                "choices": ["fallen", "rotten", "truncated", "hollow"],
            }
        ),
    ] = "fallen"
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
    deform_noise_scale: Annotated[
        float, Field(ge=10.0, le=15.0, json_schema_extra={"editable": True})
    ]


class DeformedTreeFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = DeformedTreeParameters
    _maker_factories = (
        FallenTreeFactory,
        RottenTreeFactory,
        TruncatedTreeFactory,
        HollowTreeFactory,
    )

    def __init__(self, factory_seed, coarse=False):
        AssetFactory.__init__(self, factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> DeformedTreeParameters:
        return DeformedTreeParameters(
            seed=seed,
            maker_draw="fallen",
            base_hue=uniform(0.02, 0.08),
            skinning_scale=uniform(0.15, 0.25),
            ring_wave_scale=uniform(10, 20),
            ring_distortion=uniform(4, 10),
            deform_noise_scale=uniform(10, 15),
        )

    def apply_parameters(
        self, params: DeformedTreeParameters, *, spawn_scope: bool = True
    ) -> None:
        self._use_fixed_spawn_draws = spawn_scope
        maker_index = {"fallen": 0, "rotten": 1, "truncated": 2, "hollow": 3}
        idx = maker_index.get(params.maker_draw, 0)
        maker_factory = self._maker_factories[idx]
        with FixedSeed(params.seed):
            maker_params = maker_factory.__new__(maker_factory)._sample_init_parameters(params.seed)
        shared = {
            "base_hue": params.base_hue,
            "skinning_scale": params.skinning_scale,
            "ring_wave_scale": params.ring_wave_scale,
            "ring_distortion": params.ring_distortion,
        }
        if "noise_scale" in maker_params.model_fields:
            shared["noise_scale"] = params.deform_noise_scale
        maker_params = maker_params.model_copy(update=shared)
        self.maker = maker_factory(self.factory_seed, self.coarse)
        self.maker.apply_parameters(maker_params, spawn_scope=spawn_scope)

    def create_asset(self, **params):
        return self.maker.create_asset(**params)
