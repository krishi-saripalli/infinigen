# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


from __future__ import annotations

from typing import Any, ClassVar

import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.utils.mesh import polygon_angles
from infinigen.assets.utils.object import join_objects
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import AssetParameters, ParameterizedAssetFactory
from infinigen.core.tagging import tag_object
from infinigen.core.util.math import FixedSeed

from .agave import AgaveMonocotFactory
from .banana import BananaMonocotFactory, TaroMonocotFactory
from .grasses import GrassesMonocotFactory, MaizeMonocotFactory, WheatMonocotFactory
from .growth import MonocotGrowthFactory
from .tussock import TussockMonocotFactory
from .veratrum import VeratrumMonocotFactory


class MonocotParameters(AssetParameters):
    pass


class MonocotFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = MonocotParameters
    max_cluster = 10

    def __init__(self, factory_seed, coarse=False, factory_method=None, grass=None):
        super(MonocotFactory, self).__init__(factory_seed, coarse)
        self._factory_method_arg = factory_method
        self._grass_arg = grass
        self.init_legacy_parameters()

    def _factory_methods(self, grass: bool | None) -> list[type]:
        grass_factory = [
            TussockMonocotFactory,
            GrassesMonocotFactory,
            WheatMonocotFactory,
            MaizeMonocotFactory,
        ]
        nongrass_factory = [
            AgaveMonocotFactory,
            BananaMonocotFactory,
            TaroMonocotFactory,
            VeratrumMonocotFactory,
        ]
        if grass is None:
            return grass_factory + nongrass_factory
        return grass_factory if grass else nongrass_factory

    def _resolve_factory_method(self, seed: int):
        methods = self._factory_methods(self._grass_arg)
        weights = np.array([1] * len(methods))
        weights = weights / weights.sum()
        if self._factory_method_arg is None:
            with FixedSeed(seed):
                return np.random.choice(methods, p=weights)
        return self._factory_method_arg

    def _sample_init_parameters(self, seed: int) -> MonocotParameters:
        self._factory_method = self._resolve_factory_method(seed)
        self._cluster_n = 1
        self._cluster_angles: tuple[float, ...] = ()
        self._cluster_radius: tuple[float, ...] = ()
        return MonocotParameters(seed=seed)

    def _is_grass_method(self, factory_method: type) -> bool:
        return factory_method in (
            TussockMonocotFactory,
            GrassesMonocotFactory,
            WheatMonocotFactory,
            MaizeMonocotFactory,
        )

    def _sample_spawn_parameters(
        self, params: MonocotParameters, seed: int, i: int
    ) -> MonocotParameters:
        if not self._is_grass_method(self._factory_method):
            return params
        n = np.random.randint(1, 6)
        angles = polygon_angles(n, np.pi / 4, np.pi * 2)
        radius = uniform(0.08, 0.16, n)
        self._cluster_n = n
        self._cluster_angles = tuple(float(a) for a in angles)
        self._cluster_radius = tuple(float(r) for r in radius)
        return params

    def apply_parameters(
        self, params: MonocotParameters, *, spawn_scope: bool = True
    ) -> None:
        if not hasattr(self, "_factory_method"):
            self._factory_method = self._resolve_factory_method(params.seed)
        self.factory: MonocotGrowthFactory = self._factory_method(
            self.factory_seed, self.coarse
        )
        self.cluster_n = getattr(self, "_cluster_n", 1)
        self.cluster_angles = getattr(self, "_cluster_angles", ())
        self.cluster_radius = getattr(self, "_cluster_radius", ())
        self._use_fixed_spawn_draws = spawn_scope

    def create_asset(self, i, **params) -> bpy.types.Object:
        params["decorate"] = True
        if self.factory.is_grass:
            if self._use_fixed_spawn_draws:
                n = self.cluster_n
                angles = self.cluster_angles
                radius = self.cluster_radius
            else:
                n = np.random.randint(1, 6)
                angles = polygon_angles(n, np.pi / 4, np.pi * 2)
                radius = uniform(0.08, 0.16, n)
            monocots = [
                self.factory.create_asset(**params, i=j + i * self.max_cluster)
                for j in range(n)
            ]
            for m, a, r in zip(monocots, angles, radius):
                m.location = r * np.cos(a), r * np.sin(a), 0
            obj = join_objects(monocots)
            tag_object(obj, "monocot")
            return obj
        m = self.factory.create_asset(**params)
        tag_object(m, "monocot")
        return m
