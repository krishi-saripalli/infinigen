# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei

from __future__ import annotations

from typing import Annotated, ClassVar

import bpy
import numpy as np
from pydantic import Field

from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import (
    AssetParameters,
    ParameterizedAssetFactory,
)

from .cantilever import CantileverStaircaseFactory
from .curved import CurvedStaircaseFactory
from .l_shaped import LShapedStaircaseFactory
from .spiral import SpiralStaircaseFactory
from .straight import StraightStaircaseFactory
from .u_shaped import UShapedStaircaseFactory

_STAIRCASE_FACTORIES = [
    StraightStaircaseFactory,
    LShapedStaircaseFactory,
    UShapedStaircaseFactory,
    SpiralStaircaseFactory,
    CurvedStaircaseFactory,
    CantileverStaircaseFactory,
]
_STAIRCASE_PROBS = np.array([4, 3, 3, 1, 2, 2], dtype=float)
_STAIRCASE_CUMPROBS = np.cumsum(_STAIRCASE_PROBS) / _STAIRCASE_PROBS.sum()


class StaircaseParameters(AssetParameters):
    staircase_type_draw: Annotated[
        str,
        Field(
            json_schema_extra={
                "editable": True,
                "kind": "enum",
                "choices": [
                    "straight",
                    "l_shaped",
                    "u_shaped",
                    "spiral",
                    "curved",
                    "cantilever",
                ],
            }
        ),
    ] = "straight"


def _resolve_staircase_factory(staircase_type_draw: str | float) -> type:
    if isinstance(staircase_type_draw, str):
        by_name = {
            "straight": StraightStaircaseFactory,
            "l_shaped": LShapedStaircaseFactory,
            "u_shaped": UShapedStaircaseFactory,
            "spiral": SpiralStaircaseFactory,
            "curved": CurvedStaircaseFactory,
            "cantilever": CantileverStaircaseFactory,
        }
        return by_name.get(staircase_type_draw, StraightStaircaseFactory)
    idx = int(np.searchsorted(_STAIRCASE_CUMPROBS, staircase_type_draw, side="right"))
    return _STAIRCASE_FACTORIES[min(idx, len(_STAIRCASE_FACTORIES) - 1)]


class StaircaseFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = StaircaseParameters
    factories = _STAIRCASE_FACTORIES
    probs = _STAIRCASE_PROBS

    def __init__(self, factory_seed, coarse=False):
        super(StaircaseFactory, self).__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> StaircaseParameters:
        return StaircaseParameters(seed=seed, staircase_type_draw="straight")

    def apply_parameters(
        self, params: StaircaseParameters, *, spawn_scope: bool = True
    ) -> None:
        factory_cls = _resolve_staircase_factory(params.staircase_type_draw)
        self.base_factory = factory_cls(self.factory_seed, self.coarse)
        self._use_fixed_spawn_draws = spawn_scope

    def create_asset(self, **params) -> bpy.types.Object:
        return self.base_factory.create_asset(**params)

    def finalize_assets(self, assets):
        self.base_factory.finalize_assets(assets)
