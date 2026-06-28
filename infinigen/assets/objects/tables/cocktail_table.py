# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Yiming Zuo: primary author
# - Alexander Raistrick: implement placeholder

from __future__ import annotations

from typing import Annotated, Any, ClassVar

import bpy
from numpy.random import choice, uniform
from pydantic import Field

from infinigen.assets.composition import material_assignments
from infinigen.assets.objects.tables.legs.single_stand import (
    nodegroup_generate_single_stand,
)
from infinigen.assets.objects.tables.legs.straight import (
    nodegroup_generate_leg_straight,
)
from infinigen.assets.objects.tables.legs.wheeled import nodegroup_wheeled_leg
from infinigen.assets.objects.tables.strechers import nodegroup_strecher
from infinigen.assets.objects.tables.table_top import nodegroup_generate_table_top
from infinigen.assets.objects.tables.table_utils import (
    nodegroup_create_anchors,
    nodegroup_create_legs_and_strechers,
)
from infinigen.core import surface, tagging
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import (
    AssetParameters,
    ParameterizedAssetFactory,
)
from infinigen.core.surface import NoApply
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import weighted_sample


@node_utils.to_nodegroup(
    "geometry_create_legs", singleton=False, type="GeometryNodeTree"
)
def geometry_create_legs(nw: NodeWrangler, **kwargs):
    createanchors = nw.new_node(
        nodegroup_create_anchors().name,
        input_kwargs={
            "Profile N-gon": kwargs["Leg Number"],
            "Profile Width": kwargs["Leg Placement Top Relative Scale"]
            * kwargs["Top Profile Width"],
            "Profile Aspect Ratio": 1.0000,
        },
    )

    if kwargs["Leg Style"] == "single_stand":
        leg = nw.new_node(
            nodegroup_generate_single_stand(**kwargs).name,
            input_kwargs={
                "Leg Height": kwargs["Leg Height"],
                "Leg Diameter": kwargs["Leg Diameter"],
                "Resolution": 64,
            },
        )

        leg = nw.new_node(
            nodegroup_create_legs_and_strechers().name,
            input_kwargs={
                "Anchors": createanchors,
                "Keep Legs": True,
                "Leg Instance": leg,
                "Table Height": kwargs["Top Height"],
                "Leg Bottom Relative Scale": kwargs[
                    "Leg Placement Bottom Relative Scale"
                ],
                "Align Leg X rot": True,
            },
        )

    elif kwargs["Leg Style"] == "straight":
        leg = nw.new_node(
            nodegroup_generate_leg_straight(**kwargs).name,
            input_kwargs={
                "Leg Height": kwargs["Leg Height"],
                "Leg Diameter": kwargs["Leg Diameter"],
                "Resolution": 32,
                "N-gon": kwargs["Leg NGon"],
                "Fillet Ratio": 0.1,
            },
        )

        strecher = nw.new_node(
            nodegroup_strecher().name,
            input_kwargs={"Profile Width": kwargs["Leg Diameter"] * 0.5},
        )

        leg = nw.new_node(
            nodegroup_create_legs_and_strechers().name,
            input_kwargs={
                "Anchors": createanchors,
                "Keep Legs": True,
                "Leg Instance": leg,
                "Table Height": kwargs["Top Height"],
                "Strecher Instance": strecher,
                "Strecher Index Increment": kwargs["Strecher Increament"],
                "Strecher Relative Position": kwargs["Strecher Relative Pos"],
                "Leg Bottom Relative Scale": kwargs[
                    "Leg Placement Bottom Relative Scale"
                ],
                "Align Leg X rot": True,
            },
        )

    elif kwargs["Leg Style"] == "wheeled":
        leg = nw.new_node(
            nodegroup_wheeled_leg(**kwargs).name,
            input_kwargs={
                "Joint Height": kwargs["Leg Joint Height"],
                "Leg Diameter": kwargs["Leg Diameter"],
                "Top Height": kwargs["Top Height"],
                "Wheel Width": kwargs["Leg Wheel Width"],
                "Wheel Rotation": kwargs["Leg Wheel Rot"],
                "Pole Length": kwargs["Leg Pole Length"],
                "Leg Number": kwargs["Leg Pole Number"],
            },
        )

    else:
        raise NotImplementedError

    leg = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": leg, "Material": kwargs["LegMaterial"]},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": leg},
        attrs={"is_active_output": True},
    )


def geometry_assemble_table(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.6.4 of the node_transpiler

    generatetabletop = nw.new_node(
        nodegroup_generate_table_top().name,
        input_kwargs={
            "Thickness": kwargs["Top Thickness"],
            "N-gon": kwargs["Top Profile N-gon"],
            "Profile Width": kwargs["Top Profile Width"],
            "Aspect Ratio": kwargs["Top Profile Aspect Ratio"],
            "Fillet Ratio": kwargs["Top Profile Fillet Ratio"],
            "Fillet Radius Vertical": kwargs["Top Vertical Fillet Ratio"],
        },
    )

    tabletop_instance = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": generatetabletop,
            "Translation": (0.0000, 0.0000, kwargs["Top Height"]),
        },
    )

    tabletop_instance = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": tabletop_instance, "Material": kwargs["TopMaterial"]},
    )

    legs = nw.new_node(geometry_create_legs(**kwargs).name)

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [tabletop_instance, legs]}
    )

    resample_curve = nw.new_node(
        Nodes.ResampleCurve, input_kwargs={"Curve": generatetabletop.outputs["Curve"]}
    )
    fill_curve = nw.new_node(Nodes.FillCurve, input_kwargs={"Curve": resample_curve})

    voff = kwargs["Top Height"] + kwargs["Top Thickness"]
    extrude_mesh = nw.new_node(
        Nodes.ExtrudeMesh,
        input_kwargs={"Mesh": fill_curve, "Offset Scale": -voff, "Individual": False},
    )
    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [extrude_mesh.outputs["Mesh"], fill_curve]},
    )
    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry_1, "Translation": (0, 0, voff)},
    )
    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            0: kwargs["is_placeholder"],
            1: join_geometry,
            2: transform_geometry_1,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": switch},
        attrs={"is_active_output": True},
    )


class TableCocktailParameters(AssetParameters):
    top_thickness: Annotated[
        float, Field(ge=0.02, le=0.05, json_schema_extra={"editable": False})
    ]
    strecher_increament: Annotated[
        int,
        Field(
            json_schema_extra={"editable": False, "kind": "enum", "choices": [0, 1, 2]}
        ),
    ] = 1


class TableCocktailFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = TableCocktailParameters

    def __init__(self, factory_seed, coarse=False, dimensions=None):
        self._dimensions = dimensions
        super(TableCocktailFactory, self).__init__(factory_seed, coarse=coarse)
        self.init_legacy_parameters()

    def _sample_materials(self, seed: int) -> tuple[dict[str, Any], Any | None, Any | None]:
        with FixedSeed(seed):
            material_params = {
                "TopMaterial": weighted_sample(material_assignments.table_top)(),
                "LegMaterial": weighted_sample(material_assignments.tableware)(),
            }
            wrapped_params = {k: v() for k, v in material_params.items()}
            scratch_prob, edge_wear_prob = material_assignments.wear_tear_prob
            scratch_fn, edge_wear_fn = material_assignments.wear_tear
            scratch = None if uniform() > scratch_prob else scratch_fn()
            edge_wear = None if uniform() > edge_wear_prob else edge_wear_fn()
            return wrapped_params, scratch, edge_wear

    def _sample_geometry_spawn_state(self, x: float) -> dict[str, Any]:
        n_gon = choice([4, 32])
        round_table = n_gon >= 32
        leg_style = choice(["straight", "single_stand"])
        if leg_style == "single_stand":
            leg_number = 1
            leg_diameter = uniform(0.7 * x, 0.9 * x)
            leg_curve_ctrl_pts = [
                (0.0, uniform(0.1, 0.2)),
                (0.5, uniform(0.1, 0.2)),
                (0.9, uniform(0.2, 0.3)),
                (1.0, 1.0),
            ]
        elif leg_style == "straight":
            leg_diameter = uniform(0.05, 0.07)
            leg_number = choice([3, 4]) if round_table else n_gon
            leg_curve_ctrl_pts = [
                (0.0, 1.0),
                (0.4, uniform(0.85, 0.95)),
                (1.0, uniform(0.4, 0.6)),
            ]
        else:
            raise NotImplementedError
        return {
            "round_table": round_table,
            "leg_style": leg_style,
            "leg_number": leg_number,
            "leg_diameter": leg_diameter,
            "leg_curve_ctrl_pts": leg_curve_ctrl_pts,
            "leg_ngon": choice([4, 32]),
            "top_profile_fillet_ratio": 0.499 if round_table else uniform(0.0, 0.05),
        }

    def _build_geometry_params(
        self, params: TableCocktailParameters, spawn: dict[str, Any]
    ) -> dict[str, Any]:
        if self._dimensions is not None:
            x, _, height = self._dimensions
        else:
            x = self._x
            height = self.height
        round_table = spawn["round_table"]
        return {
            "Top Profile N-gon": 32 if round_table else 4,
            "Top Profile Width": x if round_table else 1.414 * x,
            "Top Profile Aspect Ratio": 1.0,
            "Top Profile Fillet Ratio": spawn["top_profile_fillet_ratio"],
            "Top Thickness": params.top_thickness,
            "Top Vertical Fillet Ratio": self.top_vertical_fillet_ratio,
            "Height": height,
            "Top Height": height - params.top_thickness,
            "Leg Number": spawn["leg_number"],
            "Leg Style": spawn["leg_style"],
            "Leg NGon": spawn["leg_ngon"],
            "Leg Placement Top Relative Scale": 0.7,
            "Leg Placement Bottom Relative Scale": self.leg_placement_bottom_relative_scale,
            "Leg Height": 1.0,
            "Leg Diameter": spawn["leg_diameter"],
            "Leg Curve Control Points": spawn["leg_curve_ctrl_pts"],
            "Strecher Relative Pos": self.strecher_relative_pos,
            "Strecher Increament": params.strecher_increament,
        }

    def _resolve_x(self, seed: int) -> float:
        if self._dimensions is not None:
            return self._dimensions[0]
        with FixedSeed(seed):
            return uniform(0.5, 0.8)

    def _sample_init_parameters(self, seed: int) -> TableCocktailParameters:
        material_params, scratch, edge_wear = self._sample_materials(seed)
        self._material_params = material_params
        self._scratch = scratch
        self._edge_wear = edge_wear
        return TableCocktailParameters(
            seed=seed,
            top_thickness=uniform(0.02, 0.05),
            strecher_increament=1,
        )

    def _sample_spawn_parameters(
        self, params: TableCocktailParameters, seed: int, i: int
    ) -> TableCocktailParameters:
        self._x = self._resolve_x(params.seed)
        self._geometry_spawn = self._sample_geometry_spawn_state(self._x)
        return params

    def apply_parameters(
        self, params: TableCocktailParameters, *, spawn_scope: bool = True
    ) -> None:
        if not hasattr(self, "_material_params"):
            material_params, scratch, edge_wear = self._sample_materials(params.seed)
            self._material_params = material_params
            self._scratch = scratch
            self._edge_wear = edge_wear
        # NOTE: top_thickness effect varies by round_table and leg_style spawn branches; excluded from quartet sampling.
        # NOTE: height, top_vertical_fillet_ratio, leg_placement_bottom_relative_scale, and strecher_relative_pos do not elicit a reliable visual change in exported geometry; sampled on self from seed, excluded from quartet sampling.
        with FixedSeed(params.seed):
            if self._dimensions is None:
                self.height = uniform(1.0, 1.5)
            self.top_vertical_fillet_ratio = uniform(0.1, 0.3)
            self.leg_placement_bottom_relative_scale = uniform(1.1, 1.3)
            self.strecher_relative_pos = uniform(0.2, 0.6)
        # NOTE: x resampled via cached _geometry_spawn in spawn path overwrote edits; sampled on self from seed, excluded from quartet sampling.
        self._x = self._resolve_x(params.seed)
        self.dimensions = self._dimensions
        if spawn_scope and hasattr(self, "_geometry_spawn"):
            spawn = self._geometry_spawn
        else:
            spawn = self._sample_geometry_spawn_state(self._x)
        self.params = {
            **self._build_geometry_params(params, spawn),
            **self._material_params,
        }
        self.clothes_scatter = NoApply()
        self.scratch = self._scratch
        self.edge_wear = self._edge_wear
        self._use_fixed_spawn_draws = spawn_scope

    def _execute_geonodes(self, is_placeholder):
        bpy.ops.mesh.primitive_plane_add(
            size=2,
            enter_editmode=False,
            align="WORLD",
            location=(0, 0, 0),
            scale=(1, 1, 1),
        )
        obj = bpy.context.active_object

        kwargs = {**self.params, "is_placeholder": is_placeholder}
        surface.add_geomod(
            obj, geometry_assemble_table, apply=True, input_kwargs=kwargs
        )
        tagging.tag_system.relabel_obj(obj)

        return obj

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        return self._execute_geonodes(is_placeholder=True)

    def create_asset(self, **_):
        return self._execute_geonodes(is_placeholder=False)

    def finalize_assets(self, assets):
        self.clothes_scatter.apply(assets)
        # if self.scratch:
        #     self.scratch.apply(assets)
        # if self.edge_wear:
        #     self.edge_wear.apply(assets)
