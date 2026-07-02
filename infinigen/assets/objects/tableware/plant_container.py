# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from __future__ import annotations

from typing import Annotated, ClassVar

import bpy
import numpy as np
from numpy.random import uniform
from pydantic import Field

from infinigen.assets.composition import material_assignments
from infinigen.assets.objects.cactus import CactusFactory
from infinigen.assets.objects.monocot import MonocotFactory
from infinigen.assets.objects.mushroom import MushroomFactory
from infinigen.assets.objects.small_plants import (
    FernFactory,
    SnakePlantFactory,
    SpiderPlantFactory,
    SucculentFactory,
)
from infinigen.assets.objects.tableware.pot import (
    _POT_TIER1_FIELDS,
    PotFactory,
    PotParameters,
)
from infinigen.assets.objects.tableware.base import (
    apply_tableware_from_draws,
    sample_tableware_base,
)
from infinigen.assets.utils.decorate import (
    read_edge_center,
    read_edge_direction,
    remove_vertices,
    select_edges,
    subsurf,
)
from infinigen.assets.utils.object import join_objects, new_bbox, origin2lowest
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import (
    AssetParameters,
    ParameterizedAssetFactory,
)
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform, weighted_sample


class PlantPotParameters(PotParameters):
    pass


# Subclassing does not inherit a parent's model_fields.pop() (pydantic
# recomputes fields for the subclass from the original annotations), so
# PotParameters's _POT_TIER1_FIELDS removal must be re-applied here too.
for _field in (*_POT_TIER1_FIELDS, "scale", "r_mid", "thickness"):
    PlantPotParameters.model_fields.pop(_field, None)
PlantPotParameters.model_rebuild(force=True)


class PlantPotFactory(PotFactory):
    parameters_model: ClassVar[type[AssetParameters]] = PlantPotParameters

    def __init__(self, factory_seed, coarse=False):
        super(PlantPotFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(self.factory_seed):
            self.has_handle = self.has_bar = self.has_guard = False

    def _sample_init_parameters(self, seed: int) -> PlantPotParameters:
        base = sample_tableware_base(seed)
        r_expand = 1 if uniform(0, 1) < 0.2 else log_uniform(1.0, 1.2)
        return PlantPotParameters(
            seed=seed,
            r_expand=r_expand,
            depth=log_uniform(0.3, 0.8),
            pot_depth=log_uniform(0.6, 2.0),
            has_bar=False,
            bar_scale_x=np.clip(log_uniform(0.6, 1.0) * log_uniform(0.6, 1.5), 0.6, 1.0),
            scratch_draw=base["scratch_draw"],
            edge_wear_draw=base["edge_wear_draw"],
        )

    def apply_parameters(
        self, params: PlantPotParameters, *, spawn_scope: bool = True
    ) -> None:
        with FixedSeed(params.seed):
            pan_scale = log_uniform(0.1, 0.15)
            # NOTE: lower_thresh is not wired to exported geometry; excluded from quartet sampling.
            lower_thresh = uniform(0.5, 0.8)
        apply_tableware_from_draws(
            self,
            seed=params.seed,
            lower_thresh=lower_thresh,
            scale=pan_scale,
            metal_color=None,
        )
        self.depth = params.depth
        self.r_expand = params.r_expand
        self.scale = pan_scale
        self.has_bar = False
        self.has_handle = False
        self.has_handle_hole = False
        self.has_guard = False
        with FixedSeed(params.seed):
            # NOTE: r_mid is not wired to exported geometry; excluded from quartet sampling.
            self.r_mid = log_uniform(1.0, 1.3)
            # NOTE: thickness is not wired to exported geometry; excluded from quartet sampling.
            self.thickness = log_uniform(0.04, 0.06)
            # NOTE: n_vertices is not wired to exported geometry; excluded from quartet sampling.
            n_factor = log_uniform(4, 8)
            n = 4 * int(n_factor)
            # NOTE: grid_offset is not wired to exported geometry; excluded from quartet sampling.
            grid_offset = int(np.random.randint(n // 4))
            # NOTE: hole_scale is not wired to exported geometry; excluded from quartet sampling.
            hole_scale = uniform(0.06, 0.1)
            # NOTE: hole_location_frac is not wired to exported geometry; excluded from quartet sampling.
            hole_location_frac = uniform(0.8, 0.9)
        self._use_fixed_spawn_draws = spawn_scope
        if spawn_scope:
            self._n = n
            self._grid_offset = grid_offset
            self._hole_scale = hole_scale
            self._hole_location_frac = hole_location_frac


class PlantContainerParameters(AssetParameters):
    dirt_ratio: Annotated[float, Field(ge=0.7, le=0.8, json_schema_extra={"editable": True})]
    top_size: Annotated[float, Field(ge=0.4, le=0.6, json_schema_extra={"editable": True})]
    pot_depth: Annotated[float, Field(ge=0.5, le=1.0, json_schema_extra={"editable": True})]
    pot_scale: Annotated[float, Field(ge=0.1, le=0.15, json_schema_extra={"editable": True})]
    pot_r_expand: Annotated[float, Field(ge=1.1, le=1.3, json_schema_extra={"editable": True})]
    pot_alpha: Annotated[float, Field(ge=0.5, le=0.8, json_schema_extra={"editable": True})]


class PlantContainerFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = PlantContainerParameters
    plant_factories = [
        CactusFactory,
        MushroomFactory,
        FernFactory,
        SucculentFactory,
        SpiderPlantFactory,
        SnakePlantFactory,
    ]

    def __init__(self, factory_seed, coarse=False):
        super(PlantContainerFactory, self).__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_plant_factory(self, seed: int) -> None:
        with FixedSeed(seed):
            fn = np.random.choice(self.plant_factories)
            self.plant_factory = fn(seed)
            self.dirt_surface = weighted_sample(material_assignments.potting_soil)()

    def _sample_init_parameters(self, seed: int) -> PlantContainerParameters:
        self._sample_plant_factory(seed)
        pot_r_expand = uniform(1.1, 1.3)
        return PlantContainerParameters(
            seed=seed,
            dirt_ratio=uniform(0.7, 0.8),
            top_size=uniform(0.4, 0.6),
            pot_depth=log_uniform(0.5, 1.0),
            pot_scale=log_uniform(0.1, 0.15),
            pot_r_expand=pot_r_expand,
            pot_alpha=uniform(0.5, 0.8),
        )

    def apply_parameters(
        self, params: PlantContainerParameters, *, spawn_scope: bool = True
    ) -> None:
        self._sample_plant_factory(params.seed)
        self.base_factory = PlantPotFactory(params.seed, self.coarse)
        self.base_factory.depth = params.pot_depth
        self.base_factory.scale = params.pot_scale
        self.base_factory.r_expand = params.pot_r_expand
        self.base_factory.r_mid = (params.pot_r_expand - 1) * params.pot_alpha + 1
        with FixedSeed(params.seed):
            self.base_factory.surface = weighted_sample(
                material_assignments.decorative_hard
            )()()
        self.dirt_ratio = params.dirt_ratio
        self.top_size = params.top_size
        self.side_size = params.pot_scale * params.pot_r_expand
        self._use_fixed_spawn_draws = spawn_scope

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        return new_bbox(
            -self.side_size,
            self.side_size,
            -self.side_size,
            self.side_size,
            -0.02,
            self.base_factory.depth * self.base_factory.scale + self.top_size,
        )

    def create_asset(self, i, **params) -> bpy.types.Object:
        obj = self.base_factory.create_asset(i=i, **params)
        horizontal = np.abs(read_edge_direction(obj)[:, -1]) < 0.1

        edge_center = read_edge_center(obj)
        z = edge_center[:, -1]
        dirt_z = self.dirt_ratio * self.base_factory.depth * self.base_factory.scale
        idx = np.argmin(np.abs(z - dirt_z) - horizontal * 10)
        radius = np.sqrt((edge_center[idx] ** 2)[:2].sum())

        selection = np.zeros_like(z).astype(bool)
        selection[idx] = True
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_mode(type="EDGE")
            select_edges(obj, selection)
            bpy.ops.mesh.loop_multi_select(ring=False)
            bpy.ops.mesh.duplicate_move()
            bpy.ops.mesh.separate(type="SELECTED")

        dirt_ = bpy.context.selected_objects[-1]
        butil.select_none()
        self.base_factory.finalize_assets(obj)
        with butil.ViewportMode(dirt_, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.fill_grid()
        subsurf(dirt_, 3)
        self.dirt_surface.apply(dirt_)
        butil.apply_modifiers(dirt_)

        remove_vertices(dirt_, lambda x, y, z: np.sqrt(x**2 + y**2) > radius * 0.92)
        dirt_.location[-1] -= 0.02

        plant = self.plant_factory.spawn_asset(i=i, loc=(0, 0, 0), rot=(0, 0, 0))
        origin2lowest(plant, approximate=True)
        self.plant_factory.finalize_assets(plant)

        scale = np.min(
            np.array([self.side_size, self.side_size, self.top_size])
            / np.max(np.abs(np.array(plant.bound_box)), 0)
        )
        plant.scale = [scale] * 3
        plant.location[-1] = dirt_z

        obj = join_objects([obj, plant, dirt_])
        return obj


class LargePlantContainerParameters(PlantContainerParameters):
    large_pot_depth: Annotated[
        float, Field(ge=1.0, le=1.5, json_schema_extra={"editable": True})
    ]
    large_pot_scale: Annotated[
        float, Field(ge=0.15, le=0.25, json_schema_extra={"editable": True})
    ]
    side_size_mult: Annotated[
        float, Field(ge=1.5, le=2.0, json_schema_extra={"editable": True})
    ]
    large_top_size: Annotated[
        float, Field(ge=1.0, le=1.5, json_schema_extra={"editable": True})
    ]


class LargePlantContainerFactory(PlantContainerFactory):
    parameters_model: ClassVar[type[AssetParameters]] = LargePlantContainerParameters
    plant_factories = [MonocotFactory]

    def __init__(self, factory_seed, coarse=False):
        AssetFactory.__init__(self, factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> LargePlantContainerParameters:
        self._sample_plant_factory(seed)
        pot_r_expand = uniform(1.1, 1.3)
        large_pot_scale = log_uniform(0.15, 0.25)
        return LargePlantContainerParameters(
            seed=seed,
            dirt_ratio=uniform(0.7, 0.8),
            top_size=uniform(0.4, 0.6),
            pot_depth=log_uniform(0.5, 1.0),
            pot_scale=log_uniform(0.1, 0.15),
            pot_r_expand=pot_r_expand,
            pot_alpha=uniform(0.5, 0.8),
            large_pot_depth=log_uniform(1.0, 1.5),
            large_pot_scale=large_pot_scale,
            side_size_mult=uniform(1.5, 2.0),
            large_top_size=uniform(1.0, 1.5),
        )

    def apply_parameters(
        self, params: LargePlantContainerParameters, *, spawn_scope: bool = True
    ) -> None:
        self._sample_plant_factory(params.seed)
        self.base_factory = PlantPotFactory(params.seed, self.coarse)
        self.base_factory.depth = params.large_pot_depth
        self.base_factory.scale = params.large_pot_scale
        self.base_factory.r_expand = params.pot_r_expand
        self.base_factory.r_mid = (params.pot_r_expand - 1) * params.pot_alpha + 1
        with FixedSeed(params.seed):
            self.base_factory.surface = weighted_sample(
                material_assignments.decorative_hard
            )()()
        self.dirt_ratio = params.dirt_ratio
        self.top_size = params.large_top_size
        self.side_size = (
            params.large_pot_scale * params.side_size_mult * params.pot_r_expand
        )
        self._use_fixed_spawn_draws = spawn_scope
