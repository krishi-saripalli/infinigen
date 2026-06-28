# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from __future__ import annotations

from collections.abc import Iterable
from functools import cached_property
from statistics import mean
from typing import Any, ClassVar

import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.objects.fruits.general_fruit import FruitFactoryGeneralFruit
from infinigen.assets.objects.tableware.bowl import BowlFactory
from infinigen.assets.objects.tableware.pot import PotFactory
from infinigen.assets.utils.decorate import read_co, write_co
from infinigen.assets.utils.misc import make_normalized_factory, subclasses
from infinigen.core.placement.factory import AssetFactory, make_asset_collection
from infinigen.core.placement.instance_scatter import scatter_instances
from infinigen.core.placement.parameters import (
    LegacyBridgeParameters,
    ParameterizedAssetFactory,
    apply_bridge_parameters,
)
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed


class FruitCover:
    def __init__(
        self,
        factory_seed=0,
        params: FruitContainerParameters | None = None,
        factory: "FruitContainerFactory | None" = None,
    ):
        self.factory = factory
        with FixedSeed(factory_seed):
            fruit_factory_fns = list(
                subclasses(FruitFactoryGeneralFruit).difference(
                    [FruitFactoryGeneralFruit]
                )
            )
            fruit_factory_fn = make_normalized_factory(
                np.random.choice(fruit_factory_fns)
            )
            self.col = make_asset_collection(
                fruit_factory_fn(np.random.randint(1e5)), name="fruit", n=5
            )
            self.dimension = mean(mean(o.dimensions) for o in self.col.objects)
            self.shrink_rate = max(self.dimension, 2.0)
        self.params = params

    def apply(self, obj, selection=None):
        if self.factory is not None and hasattr(self.factory, "fruit_scale"):
            fruit_scale = self.factory.fruit_scale
        elif self.params is not None and hasattr(self.params, "fruit_scale"):
            fruit_scale = self.params.fruit_scale
        else:
            fruit_scale = uniform(0.06, 0.08)
        if self.factory is not None and hasattr(self.factory, "spacing_factor"):
            spacing_factor = self.factory.spacing_factor
            scale_rand = self.factory.scale_rand
        else:
            spacing_factor = uniform(0.5, 0.7)
            scale_rand = uniform(0.1, 0.3)
        for obj in obj if isinstance(obj, Iterable) else [obj]:
            scale = fruit_scale / self.shrink_rate
            scattered = scatter_instances(
                base_obj=obj,
                collection=self.col,
                density=1e3,
                min_spacing=scale * self.dimension * spacing_factor,
                scale=scale,
                scale_rand=scale_rand,
                selection=selection,
                ground_offset=self.dimension * 0.2 * scale,
                apply_geo=True,
                realize=True,
            )
            scattered.parent = obj


def _fruit_container_legacy_init(inst: Any, seed: int, coarse: bool) -> None:
    base_factory_fns = [BowlFactory, PotFactory]
    probs = np.array([1, 1])
    base_factory_fn = np.random.choice(base_factory_fns, p=probs / probs.sum())
    inst.base_factory = base_factory_fn(seed, coarse)
    inst.cover_seed = seed


class FruitContainerParameters(LegacyBridgeParameters):
    pass


class FruitContainerFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[LegacyBridgeParameters]] = FruitContainerParameters

    def __init__(self, factory_seed, coarse=False):
        super(FruitContainerFactory, self).__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> FruitContainerParameters:
        with FixedSeed(seed):
            inst = FruitContainerFactory.__new__(FruitContainerFactory)
            AssetFactory.__init__(inst, seed, self.coarse)
            _fruit_container_legacy_init(inst, seed, self.coarse)
            data = {
                k: v
                for k, v in vars(inst).items()
                if k not in ("factory_seed", "coarse")
            }
            return FruitContainerParameters(seed=seed, **data)

    def apply_parameters(
        self, params: FruitContainerParameters, *, spawn_scope: bool = True
    ) -> None:
        apply_bridge_parameters(self, params, spawn_scope=spawn_scope)
        if spawn_scope:
            self._fruit_container_params = params
        # NOTE: fruit_scale and base_lower do not elicit a clear visual change in exported geometry; excluded from quartet sampling.
        with FixedSeed(params.seed):
            self.fruit_scale = uniform(0.06, 0.08)
            self.base_lower = uniform(0.005, 0.015)
        # NOTE: rim_raise, spacing_factor, and scale_rand do not elicit a clear visual change in exported geometry; excluded from quartet sampling.
        with FixedSeed(params.seed):
            self.rim_raise = uniform(0.03, 0.07)
            self.spacing_factor = uniform(0.5, 0.7)
            self.scale_rand = uniform(0.1, 0.3)
        if hasattr(self, "cover"):
            del self.__dict__["cover"]

    @cached_property
    def cover(self):
        params = getattr(self, "_fruit_container_params", None)
        if self._use_fixed_spawn_draws:
            return FruitCover(self.cover_seed, params, self)
        return FruitCover(self.cover_seed, factory=self)

    def create_placeholder(self, **params):
        box = self.base_factory.create_placeholder(**params)
        rim_raise = getattr(self, "rim_raise", 0.05)
        base_lower = getattr(self, "base_lower", 0.01)
        co = read_co(box)
        co[co[:, -1] > 0.02, -1] += rim_raise
        co[co[:, -1] < 0.02, -1] -= base_lower
        write_co(box, co)
        butil.apply_transform(box)
        return box

    def create_asset(self, **params) -> bpy.types.Object:
        return self.base_factory.create_asset(**params)

    def finalize_assets(self, assets):
        self.base_factory.finalize_assets(assets)
        self.cover.apply(assets, selection="lower_inside")
