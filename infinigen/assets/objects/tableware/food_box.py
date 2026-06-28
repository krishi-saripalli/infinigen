# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from __future__ import annotations

from typing import ClassVar

import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.materials import text
from infinigen.assets.utils.object import new_cube
from infinigen.assets.utils.uv import wrap_six_sides
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import AssetParameters, ParameterizedAssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform


class FoodBoxParameters(AssetParameters):
    pass


class FoodBoxFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = FoodBoxParameters

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_texture_shared(self, seed: int) -> bool:
        # NOTE: texture_shared_draw is sampled on self in apply_parameters; excluded from quartet sampling (material-only, not exported geometry).
        with FixedSeed(seed):
            return uniform() < 0.4

    def _sample_init_parameters(self, seed: int) -> FoodBoxParameters:
        dimensions = np.sort(log_uniform(0.05, 0.3, 3)).tolist()
        self._dimensions = np.array([dimensions[1], dimensions[0], dimensions[2]])
        self._surface = text.Text(seed)()
        self.texture_shared = self._sample_texture_shared(seed)
        return FoodBoxParameters(seed=seed)

    def apply_parameters(
        self, params: FoodBoxParameters, *, spawn_scope: bool = True
    ) -> None:
        if not hasattr(self, "_dimensions"):
            dimensions = np.sort(log_uniform(0.05, 0.3, 3)).tolist()
            self._dimensions = np.array([dimensions[1], dimensions[0], dimensions[2]])
            self._surface = text.Text(params.seed)()
        self.dimensions = self._dimensions
        self.surface = self._surface
        self.texture_shared = self._sample_texture_shared(params.seed)
        self._use_fixed_spawn_draws = spawn_scope

    def create_placeholder(self, **params):
        obj = new_cube()
        obj.scale = self.dimensions / 2
        butil.apply_transform(obj)
        return obj

    def create_asset(self, placeholder, **params) -> bpy.types.Object:
        obj = butil.copy(placeholder)
        wrap_six_sides(obj, self.surface, self.texture_shared)
        butil.modify_mesh(obj, "BEVEL", width=0.001)
        return obj
