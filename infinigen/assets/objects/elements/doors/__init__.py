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

from .casing import DoorCasingFactory
from .lite import LiteDoorFactory
from .louver import LouverDoorFactory
from .panel import GlassPanelDoorFactory, PanelDoorFactory


def random_door_factory():
    door_factories = [
        PanelDoorFactory,
        GlassPanelDoorFactory,
        LouverDoorFactory,
        LiteDoorFactory,
    ]
    door_probs = np.array([4, 2, 3, 3])
    return np.random.choice(door_factories, p=door_probs / door_probs.sum())


_DOOR_FACTORIES = [
    PanelDoorFactory,
    GlassPanelDoorFactory,
    LouverDoorFactory,
    LiteDoorFactory,
]
_DOOR_PROBS = np.array([4, 2, 3, 3], dtype=float)
_DOOR_CUMPROBS = np.cumsum(_DOOR_PROBS) / _DOOR_PROBS.sum()


class DoorParameters(AssetParameters):
    # NOTE: selects door factory type; geometry only changes when sweep crosses factory boundary.
    door_type_draw: Annotated[
        str,
        Field(
            json_schema_extra={
                "editable": False,
                "kind": "enum",
                "choices": ["panel", "glass_panel", "louver", "lite"],
            }
        ),
    ] = "panel"


def _resolve_door_factory(door_type_draw: str | float) -> type:
    if isinstance(door_type_draw, str):
        by_name = {
            "panel": PanelDoorFactory,
            "glass_panel": GlassPanelDoorFactory,
            "louver": LouverDoorFactory,
            "lite": LiteDoorFactory,
        }
        return by_name.get(door_type_draw, PanelDoorFactory)
    idx = int(np.searchsorted(_DOOR_CUMPROBS, door_type_draw, side="right"))
    return _DOOR_FACTORIES[min(idx, len(_DOOR_FACTORIES) - 1)]


class DoorFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = DoorParameters

    def __init__(self, factory_seed, coarse=False, constants=None):
        self._constants = constants
        super(DoorFactory, self).__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> DoorParameters:
        return DoorParameters(seed=seed, door_type_draw="panel")

    def apply_parameters(
        self, params: DoorParameters, *, spawn_scope: bool = True
    ) -> None:
        factory_cls = _resolve_door_factory(params.door_type_draw)
        self.base_factory = factory_cls(
            self.factory_seed, self.coarse, self._constants
        )
        self._use_fixed_spawn_draws = spawn_scope

    def create_asset(self, **params) -> bpy.types.Object:
        return self.base_factory.create_asset(**params)

    def finalize_assets(self, assets):
        self.base_factory.finalize_assets(assets)
