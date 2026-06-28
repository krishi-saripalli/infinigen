# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick, Yiming Zuo, Alejandro Newell, Lingjie Mei


from __future__ import annotations

import logging
from typing import Annotated, ClassVar

import bpy
import gin
import numpy as np
from numpy.random import uniform
from pydantic import Field

from infinigen.assets.composition import material_assignments
from infinigen.assets.objects.cloud import CloudFactory
from infinigen.assets.objects.fruits import (
    apple,
    blackberry,
    coconutgreen,
    compositional_fruit,
    durian,
    starfruit,
    strawberry,
)
from infinigen.assets.objects.leaves import (
    leaf,
    leaf_broadleaf,
    leaf_ginko,
    leaf_maple,
    leaf_pine,
    leaf_v2,
)
from infinigen.assets.objects.trees import branch, tree, treeconfigs
from infinigen.assets.utils.misc import toggle_hide, toggle_show
from infinigen.core import surface
from infinigen.core.placement import detail
from infinigen.core.placement.factory import AssetFactory, make_asset_collection
from infinigen.core.placement.parameters import (
    AssetParameters,
    ParameterizedAssetFactory,
)
from infinigen.core.placement.split_in_view import split_inview
from infinigen.core.tagging import tag_object
from infinigen.core.util import blender as butil
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import weighted_sample

from . import tree_flower

logger = logging.getLogger(__name__)


class GenericTreeParameters(AssetParameters):
    tree_scale: Annotated[
        float, Field(ge=0.25, le=0.45, json_schema_extra={"editable": True})
    ]
    max_radius: Annotated[
        float, Field(ge=0.15, le=0.25, json_schema_extra={"editable": True})
    ]
    min_radius: Annotated[
        float, Field(ge=0.01, le=0.04, json_schema_extra={"editable": True})
    ]
    branch_exponent: Annotated[
        float, Field(ge=1.5, le=2.5, json_schema_extra={"editable": True})
    ]
    child_density: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    child_max_scale: Annotated[
        float, Field(ge=1.0, le=1.5, json_schema_extra={"editable": True})
    ]


@gin.configurable
class GenericTreeFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = GenericTreeParameters
    scale = 0.35

    def __init__(
        self,
        factory_seed,
        genome: tree.TreeParams | None = None,
        child_col=None,
        trunk_surface=None,
        realize=False,
        meshing_cameras=None,
        cam_meshing_max_dist=1e7,
        coarse_mesh_placeholder=False,
        adapt_mesh_method="remesh",
        decimate_placeholder_levels=0,
        min_dist=None,
        coarse=False,
    ):
        self._tree_kwargs = {
            "realize": realize,
            "meshing_cameras": meshing_cameras,
            "cam_meshing_max_dist": cam_meshing_max_dist,
            "coarse_mesh_placeholder": coarse_mesh_placeholder,
            "adapt_mesh_method": adapt_mesh_method,
            "decimate_placeholder_levels": decimate_placeholder_levels,
            "min_dist": min_dist,
        }
        self._external_genome = genome
        self._external_child_col = child_col
        self._external_trunk_surface = trunk_surface
        super(GenericTreeFactory, self).__init__(factory_seed, coarse=coarse)
        if genome is not None:
            self._configure_from_genome(genome, child_col, trunk_surface)
        else:
            self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> GenericTreeParameters:
        self._trunk_surface = weighted_sample(material_assignments.bark)
        return GenericTreeParameters(
            seed=seed,
            tree_scale=uniform(0.25, 0.45),
            max_radius=uniform(0.15, 0.25),
            min_radius=uniform(0.01, 0.04),
            branch_exponent=uniform(1.5, 2.5),
            child_density=uniform(0.0, 1.0),
            child_max_scale=uniform(1.0, 1.5),
        )

    def apply_parameters(
        self, params: GenericTreeParameters, *, spawn_scope: bool = True
    ) -> None:
        if not hasattr(self, "_trunk_surface"):
            self._trunk_surface = weighted_sample(material_assignments.bark)
        self.scale = params.tree_scale
        self.trunk_surface = self._trunk_surface
        self._use_fixed_spawn_draws = spawn_scope
        with FixedSeed(params.seed):
            (genome, _, _), _ = random_species()
        genome.skinning.update(
            {
                "Max radius": params.max_radius,
                "Min radius": params.min_radius,
                "Exponent": params.branch_exponent,
            }
        )
        if genome.child_placement is not None:
            genome.child_placement.update(
                {
                    "Density": params.child_density,
                    "Max scale": params.child_max_scale,
                }
            )
        self.genome = genome
        self.child_col = self._external_child_col

    def _run_post_init(self) -> None:
        if self._external_genome is not None:
            return
        self._configure_from_genome(
            self.genome, self._external_child_col, self._trunk_surface
        )

    def _configure_from_genome(self, genome, child_col, trunk_surface) -> None:
        self.genome = genome
        self.child_col = child_col
        self.trunk_surface = trunk_surface
        kwargs = self._tree_kwargs
        self.realize = kwargs["realize"]
        self.cameras = kwargs["meshing_cameras"]
        self.cam_meshing_max_dist = kwargs["cam_meshing_max_dist"]
        self.adapt_mesh_method = kwargs["adapt_mesh_method"]
        self.decimate_placeholder_levels = kwargs["decimate_placeholder_levels"]
        self.coarse_mesh_placeholder = kwargs["coarse_mesh_placeholder"]
        self.min_dist = kwargs["min_dist"]

    def create_placeholder(self, i, loc, rot):
        logger.debug("generating tree skeleton")
        skeleton_obj = tree.tree_skeleton(
            self.genome.skeleton,
            self.genome.trunk_spacecol,
            self.genome.roots_spacecol,
            init_pos=(0, 0, 0),
            scale=self.scale,
        )

        if self.coarse_mesh_placeholder:
            pholder = self._create_coarse_mesh(skeleton_obj)
        else:
            pholder = butil.spawn_cube(size=4)

        butil.parent_to(skeleton_obj, pholder, no_inverse=True)
        return pholder

    def _create_coarse_mesh(self, skeleton_obj):
        logger.debug("generating skinned mesh")
        coarse_mesh = deep_clone_obj(skeleton_obj)
        surface.add_geomod(
            coarse_mesh,
            tree.skin_tree,
            input_kwargs={"params": self.genome.skinning},
            apply=True,
        )

        if self.decimate_placeholder_levels > 0:
            butil.modify_mesh(
                coarse_mesh,
                "DECIMATE",
                decimate_type="UNSUBDIV",
                iterations=self.decimate_placeholder_levels,
            )

        return coarse_mesh

    def finalize_placeholders(self, placeholders):
        if not self.coarse_mesh_placeholder:
            return
        with FixedSeed(self.factory_seed):
            logger.debug(f"adding {self.trunk_surface} to {len(placeholders)=}")
            self.trunk_surface.apply(placeholders)

    def asset_parameters(self, distance: float, vis_distance: float) -> dict:
        if self.min_dist is not None and distance < self.min_dist:
            logger.warn(
                f"{self} recieved {distance=} which violates {self.min_dist=}. Ignoring"
            )
            distance = self.min_dist
        return dict(face_size=detail.target_face_size(distance), distance=distance)

    def create_asset(
        self, placeholder, face_size, distance, **kwargs
    ) -> bpy.types.Object:
        skeleton_obj = placeholder.children[0]

        if not self.coarse_mesh_placeholder:
            skin_obj = self._create_coarse_mesh(skeleton_obj)
            self.trunk_surface.apply(self, skin_obj)
            butil.parent_to(skeleton_obj, skin_obj, no_inverse=True)
        else:
            skin_obj = butil.deep_clone_obj(placeholder)

        if self.child_col is not None:
            assert self.genome.child_placement is not None

            max_needed_child_fs = (
                detail.target_face_size(self.min_dist, global_multiplier=1)
                if self.min_dist is not None
                else None
            )

            logger.debug(f"adding tree children using {self.child_col=}")
            butil.select_none()
            surface.add_geomod(
                skeleton_obj,
                tree.add_tree_children,
                input_kwargs=dict(
                    child_col=self.child_col,
                    params=self.genome.child_placement,
                    realize=self.realize,
                    merge_dist=max_needed_child_fs,
                ),
            )

        if self.cameras is not None and distance < self.cam_meshing_max_dist:
            assert self.adapt_mesh_method != "remesh"

            skin_obj_cleanup = skin_obj
            skin_obj, outofview, vert_dists, _ = split_inview(
                skin_obj, cameras=self.cameras, vis_margin=0.15
            )
            butil.parent_to(outofview, skin_obj, no_inverse=True, no_transform=True)

            butil.delete(skin_obj_cleanup)
            face_size = detail.target_face_size(vert_dists.min())

        skin_obj.hide_render = False

        if self.adapt_mesh_method == "remesh":
            butil.modify_mesh(
                skin_obj, "SUBSURF", levels=self.decimate_placeholder_levels + 1
            )  # one extra level to smooth things out or remesh is jaggedy

        with butil.DisableModifiers(skin_obj):
            detail.adapt_mesh_resolution(
                skin_obj, face_size, method=self.adapt_mesh_method, apply=True
            )

        butil.parent_to(skin_obj, placeholder, no_inverse=True, no_transform=True)

        if self.realize:
            logger.debug("realizing tree children")
            butil.apply_modifiers(skin_obj)
            butil.apply_modifiers(skeleton_obj)

            butil.join_objects([skin_obj, skeleton_obj])
            assert len(skin_obj.children) == 0
        else:
            butil.parent_to(skeleton_obj, skin_obj, no_inverse=True)

        tag_object(skin_obj, "tree")
        butil.apply_modifiers(skin_obj)

        return skin_obj


@gin.configurable
def random_season(weights=None):
    options = ["autumn", "summer", "spring", "winter"]

    if weights is not None:
        weights = np.array([weights[k] for k in options])
    else:
        weights = np.array([0.25, 0.3, 0.4, 0.1])
    return np.random.choice(options, p=weights / weights.sum())


@gin.configurable
def random_species(season="summer", pine_chance=0.0):
    tree_species_code = np.random.rand(32)

    if season is None:
        season = random_season()

    if tree_species_code[-1] < pine_chance:
        return treeconfigs.pine_tree(), "leaf_pine"
    # elif tree_species_code < 0.2:
    #     tree_args = treeconfigs.palm_tree()
    # elif tree_species_code < 0.3:
    #     tree_args = treeconfigs.baobab_tree()
    else:
        return treeconfigs.random_tree(tree_species_code, season), None


def random_tree_child_factory(
    seed, leaf_params, leaf_type, season, apply_leaf_width=False, **kwargs
):
    if season is None:
        season = random_season()

    fruit_scale = 0.2

    leaf_width = None
    if apply_leaf_width and leaf_params is not None:
        leaf_width = leaf_params.get("leaf_width")

    if leaf_type is None:
        return None, None
    elif leaf_type == "leaf_pine":
        return leaf_pine.LeafFactoryPine(seed, season, **kwargs), None
    elif leaf_type == "leaf_ginko":
        return leaf_ginko.LeafFactoryGinko(seed, season, **kwargs), None
    elif leaf_type == "leaf_maple":
        return leaf_maple.LeafFactoryMaple(seed, season, **kwargs), None
    elif leaf_type == "leaf_broadleaf":
        return leaf_broadleaf.LeafFactoryBroadleaf(seed, season, **kwargs), None
    elif leaf_type == "leaf_v2":
        return leaf_v2.LeafFactoryV2(seed, leaf_width=leaf_width, **kwargs), None
    elif leaf_type == "berry":
        return leaf.BerryFactory(seed, leaf_params, **kwargs), None
    elif leaf_type == "apple":
        return apple.FruitFactoryApple(seed, scale=fruit_scale, **kwargs), None
    elif leaf_type == "blackberry":
        return blackberry.FruitFactoryBlackberry(
            seed, scale=fruit_scale, **kwargs
        ), None
    elif leaf_type == "coconutgreen":
        return coconutgreen.FruitFactoryCoconutgreen(
            seed, scale=fruit_scale, **kwargs
        ), None
    elif leaf_type == "durian":
        return durian.FruitFactoryDurian(seed, scale=fruit_scale, **kwargs), None
    elif leaf_type == "starfruit":
        return starfruit.FruitFactoryStarfruit(seed, scale=fruit_scale, **kwargs), None
    elif leaf_type == "strawberry":
        return strawberry.FruitFactoryStrawberry(
            seed, scale=fruit_scale, **kwargs
        ), None
    elif leaf_type == "compositional_fruit":
        return compositional_fruit.FruitFactoryCompositional(
            seed, scale=fruit_scale, **kwargs
        ), None
    elif leaf_type == "flower":
        return tree_flower.TreeFlowerFactory(
            seed, rad=uniform(0.15, 0.25), leaf_width=leaf_width, **kwargs
        ), None
    elif leaf_type == "cloud":
        return CloudFactory(seed), None
    else:
        raise ValueError(f"Unrecognized {leaf_type=}")


def make_leaf_collection(
    seed,
    leaf_params,
    n_leaf,
    leaf_types,
    decimate_rate=0.0,
    season=None,
    apply_leaf_width=False,
):
    logger.debug(f"Starting make_leaf_collection({seed=}, {n_leaf=} ...)")

    if season is None:
        season = random_season()

    weights = []

    if not isinstance(leaf_types, list):
        leaf_types = [leaf_types]

    child_factories = []
    for leaf_type in leaf_types:
        if leaf_type is not None:
            leaf_factory, _ = random_tree_child_factory(
                seed,
                leaf_params,
                leaf_type=leaf_type,
                season=season,
                apply_leaf_width=apply_leaf_width,
            )
            child_factories.append(leaf_factory)
            weights.append(1.0)

    weights = np.array(weights)
    weights /= np.sum(weights)  # normalize to 1

    col = make_asset_collection(child_factories, n_leaf, verbose=True, weights=weights)
    # if leaf_surface is not None:
    #     leaf_surface.apply(list(col.objects))
    toggle_show(col)
    for obj in col.objects:
        if decimate_rate > 0:
            butil.modify_mesh(obj, "DECIMATE", ratio=1.0 - decimate_rate, apply=True)
        butil.apply_transform(obj, rot=True, scale=True)
        butil.apply_modifiers(obj)
    toggle_hide(col)
    return col


def random_leaf_collection(season, n=5):
    (_, _, leaf_params), leaf_type = random_species(season=season)
    return make_leaf_collection(
        np.random.randint(1e5),
        leaf_params,
        n_leaf=n,
        leaf_types=leaf_type or "leaf_v2",
        decimate_rate=0.97,
    )


def make_twig_collection(
    seed,
    twig_params,
    leaf_params,
    trunk_surface,
    n_leaf,
    n_twig,
    leaf_types,
    season=None,
    twig_valid_dist=6,
    apply_leaf_width=False,
):
    logger.debug(f"Starting make_twig_collection({seed=}, {n_leaf=}, {n_twig=}...)")

    if season is None:
        season = random_season()

    if leaf_types is not None:
        child_col = make_leaf_collection(
            seed,
            leaf_params,
            n_leaf,
            leaf_types,
            season=season,
            decimate_rate=0.97,
            apply_leaf_width=apply_leaf_width,
        )
    else:
        child_col = None

    twig_factory = GenericTreeFactory(
        seed, twig_params, child_col, trunk_surface=trunk_surface, realize=True
    )
    col = make_asset_collection(
        twig_factory, n_twig, verbose=False, distance=twig_valid_dist
    )

    if child_col is not None:
        child_col.hide_viewport = False
        butil.delete(list(child_col.objects))
    return col


def make_branch_collection(seed, twig_col, fruit_col, n_branch, coarse=False):
    logger.debug(f"Starting make_branch_collection({seed=}, ...)")

    branch_factory = branch.BranchFactory(
        seed, twig_col=twig_col, fruit_col=fruit_col, coarse=coarse
    )
    col = make_asset_collection(branch_factory, n_branch, verbose=False)

    return col


class TreeParameters(GenericTreeParameters):
    leaf_width: Annotated[
        float, Field(ge=0.1, le=0.6, json_schema_extra={"editable": True})
    ]
    alpha: Annotated[float, Field(ge=0.0, le=0.3, json_schema_extra={"editable": True})]
    fruit_chance_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]
    twig_density: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": True})
    ]


@gin.configurable
class TreeFactory(GenericTreeFactory):
    parameters_model: ClassVar[type[AssetParameters]] = TreeParameters
    n_leaf = 5
    n_twig = 2

    @staticmethod
    def get_leaf_type(season):
        # return np.random.choice(['leaf', 'leaf_v2', 'flower', 'berry', 'leaf_ginko'], p=[0, 0.70, 0.15, 0, 0.15])
        # return
        # return 'leaf_maple'
        leaf_type = np.random.choice(
            ["leaf_v2", "leaf_broadleaf", "leaf_ginko", "leaf_maple"],
            p=[0.0, 0.70, 0.15, 0.15],
        )
        flower_type = np.random.choice(["flower", "berry", None], p=[1.0, 0.0, 0.0])
        if season == "spring":
            return [flower_type]
        else:
            return [leaf_type]
        # return [leaf_type, flower_type]
        # return ['leaf_broadleaf', 'leaf_maple', 'leaf_ginko', 'flower']

    @staticmethod
    def get_fruit_type():
        # return np.random.choice(['leaf', 'leaf_v2', 'flower', 'berry', 'leaf_ginko'], p=[0, 0.70, 0.15, 0, 0.15])
        # return
        # return 'leaf_maple'
        fruit_type = np.random.choice(
            [
                "apple",
                "blackberry",
                "coconutgreen",
                "durian",
                "starfruit",
                "strawberry",
                "compositional_fruit",
            ],
            p=[0.2, 0.0, 0.2, 0.2, 0.2, 0.0, 0.2],
        )

        return fruit_type

    def __init__(self, seed, season=None, coarse=False, fruit_chance=1.0, **kwargs):
        self._season = season
        self._fruit_chance = fruit_chance
        self._tree_kwargs = {
            "realize": kwargs.get("realize", False),
            "meshing_cameras": kwargs.get("meshing_cameras"),
            "cam_meshing_max_dist": kwargs.get("cam_meshing_max_dist", 1e7),
            "coarse_mesh_placeholder": kwargs.get("coarse_mesh_placeholder", False),
            "adapt_mesh_method": kwargs.get("adapt_mesh_method", "remesh"),
            "decimate_placeholder_levels": kwargs.get("decimate_placeholder_levels", 0),
            "min_dist": kwargs.get("min_dist"),
        }
        AssetFactory.__init__(self, seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> TreeParameters:
        if self._season is None:
            with FixedSeed(seed):
                self._season = str(
                    np.random.choice(["summer", "winter", "autumn", "spring"])
                )
        self._trunk_surface = weighted_sample(material_assignments.bark)
        with FixedSeed(seed):
            (_, twig_params, leaf_params), leaf_type = random_species(self._season)
            leaf_type = leaf_type or self.get_leaf_type(self._season)
            if not isinstance(leaf_type, list):
                leaf_type = [leaf_type]
            self._leaf_type = leaf_type
            self._twig_params = twig_params
            self._leaf_params = leaf_params
        twig_density = float(twig_params.child_placement["Density"])
        return TreeParameters(
            seed=seed,
            tree_scale=uniform(0.25, 0.45),
            max_radius=uniform(0.15, 0.25),
            min_radius=uniform(0.01, 0.04),
            branch_exponent=uniform(1.5, 2.5),
            child_density=uniform(0.0, 1.0),
            child_max_scale=uniform(1.0, 1.5),
            leaf_width=float(leaf_params["leaf_width"]),
            alpha=float(leaf_params["alpha"]),
            fruit_chance_draw=uniform(),
            twig_density=twig_density,
        )

    def apply_parameters(
        self, params: TreeParameters, *, spawn_scope: bool = True
    ) -> None:
        if not hasattr(self, "_trunk_surface"):
            self._trunk_surface = weighted_sample(material_assignments.bark)
        if not hasattr(self, "_season") or self._season is None:
            with FixedSeed(params.seed):
                self._season = str(
                    np.random.choice(["summer", "winter", "autumn", "spring"])
                )
        self.scale = params.tree_scale
        self.trunk_surface = self._trunk_surface
        self._use_fixed_spawn_draws = spawn_scope
        with FixedSeed(params.seed):
            (tree_params, twig_params, leaf_params), leaf_type = random_species(
                self._season
            )
            leaf_type = leaf_type or self.get_leaf_type(self._season)
            if not isinstance(leaf_type, list):
                leaf_type = [leaf_type]
            self._leaf_type = leaf_type
        tree_params.skinning.update(
            {
                "Max radius": params.max_radius,
                "Min radius": params.min_radius,
                "Exponent": params.branch_exponent,
            }
        )
        if tree_params.child_placement is not None:
            tree_params.child_placement.update(
                {
                    "Density": params.child_density,
                    "Max scale": params.child_max_scale,
                }
            )
        self.genome = tree_params
        leaf_params["leaf_width"] = params.leaf_width
        leaf_params["alpha"] = params.alpha
        self._leaf_params = leaf_params
        twig_params.child_placement["Density"] = params.twig_density
        self._twig_params = twig_params
        with FixedSeed(params.seed):
            self._fruit_type = (
                self.get_fruit_type()
                if params.fruit_chance_draw < self._fruit_chance
                else None
            )

    def _run_post_init(self) -> None:
        GenericTreeFactory._configure_from_genome(
            self, self.genome, None, self.trunk_surface
        )
        for key, value in self._tree_kwargs.items():
            if key == "realize":
                self.realize = value
            elif key == "meshing_cameras":
                self.cameras = value
            elif key == "cam_meshing_max_dist":
                self.cam_meshing_max_dist = value
            elif key == "coarse_mesh_placeholder":
                self.coarse_mesh_placeholder = value
            elif key == "adapt_mesh_method":
                self.adapt_mesh_method = value
            elif key == "decimate_placeholder_levels":
                self.decimate_placeholder_levels = value
            elif key == "min_dist":
                self.min_dist = value
        with FixedSeed(self.factory_seed):
            colname = f"assets:{self}.twigs"
            use_cached = colname in bpy.data.collections
            if use_cached == self.coarse:
                logger.warning(
                    f"In {self}, encountered {use_cached=} yet {self.coarse=}, unexpected since twigs are typically generated only in coarse"
                )
            if colname not in bpy.data.collections:
                twig_col = make_twig_collection(
                    self.factory_seed,
                    self._twig_params,
                    self._leaf_params,
                    self.trunk_surface,
                    self.n_leaf,
                    self.n_twig,
                    self._leaf_type,
                    season=self._season,
                )
                if self._fruit_type is not None:
                    fruit_col = make_leaf_collection(
                        self.factory_seed,
                        self._leaf_params,
                        self.n_leaf,
                        self._fruit_type,
                        season=self._season,
                        decimate_rate=0.0,
                    )
                else:
                    fruit_col = butil.get_collection("Empty", reuse=True)
                self.child_col = make_branch_collection(
                    self.factory_seed, twig_col, fruit_col, n_branch=self.n_twig
                )
                self.child_col.name = colname
                assert (
                    self.child_col.name == colname
                ), f"Blender truncated {colname} to {self.child_col.name}"
            else:
                self.child_col = bpy.data.collections[colname]


@gin.configurable
class BushParameters(AssetParameters):
    shrub_shape: Annotated[int, Field(ge=0, le=1, json_schema_extra={"editable": True})]
    alpha: Annotated[float, Field(ge=0.0, le=0.3, json_schema_extra={"editable": False})]
    leaf_width: Annotated[
        float, Field(ge=0.1, le=0.6, json_schema_extra={"editable": True})
    ]
    pts: Annotated[int, Field(ge=0, le=99, json_schema_extra={"editable": True})]
    leaf_type: Annotated[
        str,
        Field(
            json_schema_extra={
                "editable": False,
                "kind": "enum",
                "choices": ["leaf_v2", "flower", "berry"],
            }
        ),
    ] = "leaf_v2"


@gin.configurable
class BushFactory(GenericTreeFactory):
    parameters_model: ClassVar[type[AssetParameters]] = BushParameters
    n_leaf = 3
    n_twig = 3
    max_distance = 50

    def __init__(self, seed, coarse=False, **kwargs):
        self._bush_kwargs = kwargs
        AssetFactory.__init__(self, seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> BushParameters:
        self._trunk_surface = weighted_sample(material_assignments.bark)
        return BushParameters(
            seed=seed,
            shrub_shape=int(np.random.randint(2)),
            alpha=float(np.random.rand() * 0.3),
            leaf_width=float(np.random.rand() * 0.5 + 0.1),
            pts=int(np.random.randint(100)),
            leaf_type=str(
                np.random.choice(["leaf_v2", "flower", "berry"], p=[0.5, 0.5, 0])
            ),
        )

    def apply_parameters(self, params: BushParameters, *, spawn_scope: bool = True) -> None:
        if not hasattr(self, "_trunk_surface"):
            self._trunk_surface = weighted_sample(material_assignments.bark)
        self.shrub_shape = params.shrub_shape
        self.alpha = params.alpha
        self.leaf_width = params.leaf_width
        self.pts = params.pts
        self.leaf_type = params.leaf_type
        self.trunk_surface = self._trunk_surface
        self._use_fixed_spawn_draws = spawn_scope

    def _run_post_init(self) -> None:
        with FixedSeed(self.factory_seed):
            tree_params, twig_params, leaf_params = treeconfigs.shrub(
                shrub_shape=self.shrub_shape
            )
            leaf_params["alpha"] = self.alpha
            leaf_params["leaf_width"] = self.leaf_width
            GenericTreeFactory.__init__(
                self,
                self.factory_seed,
                tree_params,
                child_col=None,
                trunk_surface=self.trunk_surface,
                coarse=self.coarse,
                **self._bush_kwargs,
            )
            colname = f"assets:{self}.twigs"
            use_cached = colname in bpy.data.collections
            if use_cached == self.coarse:
                logger.warning(
                    f"In {self}, encountered {use_cached=} yet {self.coarse=}, unexpected since twigs are typically generated only in coarse"
                )
            if colname not in bpy.data.collections:
                self.child_col = make_twig_collection(
                    self.factory_seed,
                    twig_params,
                    leaf_params,
                    self.trunk_surface,
                    self.n_leaf,
                    self.n_twig,
                    self.leaf_type,
                    apply_leaf_width=True,
                )
                self.child_col.name = colname
                assert (
                    self.child_col.name == colname
                ), f"Blender truncated {colname} to {self.child_col.name}"
            else:
                self.child_col = bpy.data.collections[colname]
