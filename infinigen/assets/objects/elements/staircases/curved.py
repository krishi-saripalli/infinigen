# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Lingjie Mei
# - Karhan Kayan: fix constants

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from numpy.random import uniform

from infinigen.assets.objects.elements.staircases.straight import (
    StraightStaircaseFactory,
    MirroredStaircaseParameters,
    _apply_straight_switch_params,
    _sample_straight_switch_params,
)
from infinigen.assets.utils.decorate import read_co, write_co
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import (
    AssetParameters,
)
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform


def _curved_legacy_init(
    inst: Any, seed: int, coarse: bool, constants: Any = None
) -> None:
    from infinigen.assets.objects.elements.staircases.straight import (
        _straight_staircase_legacy_init,
    )

    inst.full_angle, inst.radius, inst.theta = 0, 0, 0
    _straight_staircase_legacy_init(inst, seed, coarse, constants)
    inst.has_spiral = True


class CurvedStaircaseParameters(MirroredStaircaseParameters):
    pass


class CurvedStaircaseFactory(StraightStaircaseFactory):
    parameters_model: ClassVar[type[AssetParameters]] = CurvedStaircaseParameters
    support_types = (
        "weighted_choice",
        (2, "single-rail"),
        (2, "double-rail"),
        (4, "side"),
        (4, "solid"),
        (4, "hole"),
    )

    handrail_types = "weighted_choice", (2, "horizontal-post"), (2, "vertical-post")

    def __init__(self, factory_seed, coarse=False, constants=None):
        self._constants_arg = constants
        AssetFactory.__init__(self, factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> CurvedStaircaseParameters:
        return CurvedStaircaseParameters(
            seed=seed, **_sample_straight_switch_params(seed)
        )

    def apply_parameters(
        self, params: CurvedStaircaseParameters, *, spawn_scope: bool = True
    ) -> None:
        with FixedSeed(params.seed):
            _curved_legacy_init(self, params.seed, self.coarse, self._constants_arg)
        _apply_straight_switch_params(self, params)
        self.support_type = "double-rail"
        self.has_rail = True
        self.is_handrail_circular = False
        # NOTE: cap bevel width/segments on the handrail are below detection, no within-pair strip diff; sampled on self, excluded from quartet sampling.
        with FixedSeed(params.seed):
            self.handrail_cap_width_ratio = uniform(0.2, 0.5)
            self.handrail_cap_segments = int(np.random.randint(4, 7))
        self._use_fixed_spawn_draws = spawn_scope

    def build_size_config(self):
        while True:
            self.full_angle = np.random.randint(1, 5) * np.pi / 2
            self.radius = log_uniform(1.5, 3.0)
            self.n = int(self.full_angle * self.radius / log_uniform(0.25, 0.35))
            if self.n >= 8:
                break
        self.step_height = self.constants.wall_height / self.n
        self.step_width = log_uniform(0.8, 1.6)
        self.step_length = self.step_height * log_uniform(0.8, 1.2)
        self.theta = self.full_angle / self.n

    def make_line(self, alpha):
        obj = super().make_line(alpha)
        x, y, z = read_co(obj).T
        y = self.radius * np.sin(self.theta * y / self.step_length)
        x = self.radius * (1 - np.cos(self.theta * y / self.step_length)) + alpha * x
        write_co(obj, np.stack([x, y, z], -1))
        return obj
