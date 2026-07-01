# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from __future__ import annotations

from typing import Annotated, Any, ClassVar

import numpy as np
from numpy.random import uniform
from pydantic import Field

from infinigen.core.constraints.constraint_language.constants import RoomConstants
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import (
    AssetParameters,
)
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform

from .panel import PanelDoorFactory, _panel_door_legacy_init


def _lite_door_legacy_init(
    inst: Any, seed: int, coarse: bool, constants: Any = None
) -> None:
    _panel_door_legacy_init(inst, seed, coarse, constants)
    r = uniform()
    subdivide_glass = False
    if r <= 1 / 6:
        dimension = 0, 1, uniform(0.4, 0.6), 1
        subdivide_glass = True
    elif r <= 1 / 3:
        dimension = 0, 1, 0, 1
        subdivide_glass = True
    elif r <= 1 / 2:
        dimension = 0, uniform(0.3, 0.4), uniform(0.4, 0.6), 1
    elif r <= 2 / 3:
        dimension = 0, uniform(0.3, 0.4), uniform(0.4, 0.6), 1
    elif r <= 5 / 6:
        dimension = 0, 1, 0, 1
    else:
        x = uniform(0.3, 0.35)
        dimension = x, 1 - x, uniform(0.7, 0.8), 1
    inst.x_min, inst.x_max, inst.y_min, inst.y_max = dimension
    if subdivide_glass:
        inst.x_subdivisions = np.random.choice([1, 3])
        inst.y_subdivisions = int(
            inst.height / inst.width * inst.x_subdivisions
        ) + np.random.randint(-1, 2)
    else:
        inst.x_subdivisions = 1
        inst.y_subdivisions = 1
    inst.has_glass = True


def _sample_lite_layout(
    seed: int, constants: Any | None
) -> tuple[float, float, float, int, int]:
    if constants is None:
        constants = RoomConstants()
    width = constants.door_width
    height = constants.door_size
    with FixedSeed(seed):
        r = uniform()
        subdivide_glass = False
        if r <= 1 / 6:
            dimension = 0.0, 1.0, uniform(0.4, 0.6), 1.0
            subdivide_glass = True
        elif r <= 1 / 3:
            dimension = 0.0, 1.0, 0.0, 1.0
            subdivide_glass = True
        elif r <= 1 / 2:
            dimension = 0.0, uniform(0.3, 0.4), uniform(0.4, 0.6), 1.0
        elif r <= 2 / 3:
            dimension = 0.0, uniform(0.3, 0.4), uniform(0.4, 0.6), 1.0
        elif r <= 5 / 6:
            dimension = 0.0, 1.0, 0.0, 1.0
        else:
            x = uniform(0.3, 0.35)
            dimension = x, 1 - x, uniform(0.7, 0.8), 1.0
        x_min, x_max, y_min, _ = dimension
        if subdivide_glass:
            x_subdivisions = int(np.random.choice([1, 3]))
            y_subdivisions = int(height / width * x_subdivisions) + np.random.randint(
                -1, 2
            )
            y_subdivisions = int(np.clip(y_subdivisions, 1, 5))
        else:
            x_subdivisions = 1
            y_subdivisions = 1
        x_min = float(np.clip(x_min, 0.0, 0.35))
        x_max = float(np.clip(x_max, 0.6, 1.0))
        y_min = float(np.clip(y_min, 0.0, 0.8))
    return x_min, x_max, y_min, x_subdivisions, y_subdivisions


class LiteDoorParameters(AssetParameters):
    y_min: Annotated[float, Field(ge=0.0, le=0.8, json_schema_extra={"editable": True})]
    x_subdivisions: Annotated[
        int, Field(ge=1, le=3, json_schema_extra={"editable": True})
    ]
    y_subdivisions: Annotated[
        int, Field(ge=1, le=5, json_schema_extra={"editable": True})
    ]


class LiteDoorFactory(PanelDoorFactory):
    parameters_model: ClassVar[type[AssetParameters]] = LiteDoorParameters

    def __init__(self, factory_seed, coarse=False, constants=None):
        self._constants = constants
        AssetFactory.__init__(self, factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> LiteDoorParameters:
        _, _, y_min, x_subdivisions, y_subdivisions = _sample_lite_layout(
            seed, self._constants
        )
        return LiteDoorParameters(
            seed=seed,
            y_min=y_min,
            x_subdivisions=x_subdivisions,
            y_subdivisions=y_subdivisions,
        )

    def apply_parameters(
        self, params: LiteDoorParameters, *, spawn_scope: bool = True
    ) -> None:
        with FixedSeed(params.seed):
            _panel_door_legacy_init(self, params.seed, self.coarse, self._constants)
            x_min, x_max, _, _, _ = _sample_lite_layout(params.seed, self._constants)
            panel_margin = log_uniform(0.08, 0.12)
            bevel_width = uniform(0.005, 0.01)
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = params.y_min
        self.y_max = 1.0
        self.panel_margin = panel_margin
        self.bevel_width = bevel_width
        self.x_subdivisions = params.x_subdivisions
        self.y_subdivisions = params.y_subdivisions
        self.has_glass = True
        self._use_fixed_spawn_draws = spawn_scope

    def make_panels(self):
        x_range = (
            np.linspace(self.x_min, self.x_max, self.x_subdivisions + 1)
            * (self.width - self.panel_margin * 2)
            + self.panel_margin
        )
        y_range = (
            np.linspace(self.y_min, self.y_max, self.y_subdivisions + 1)
            * (self.height - self.panel_margin * 2)
            + self.panel_margin
        )
        panels = []
        for x_min, x_max in zip(x_range[:-1], x_range[1:]):
            for y_min, y_max in zip(y_range[:-1], y_range[1:]):
                panels.append(
                    {
                        "dimension": (x_min, x_max, y_min, y_max),
                        "func": self.bevel,
                        "attribute_name": "glass",
                    }
                )
        return panels
