# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo

from __future__ import annotations

from typing import Annotated, Any, ClassVar

import bpy
from numpy.random import choice, uniform
from pydantic import Field

from infinigen.assets.composition import material_assignments
from infinigen.assets.objects.tables.legs.single_stand import (
    nodegroup_generate_single_stand,
)
from infinigen.assets.objects.tables.legs.square import nodegroup_generate_leg_square
from infinigen.assets.objects.tables.legs.straight import (
    nodegroup_generate_leg_straight,
)
from infinigen.assets.objects.tables.strechers import nodegroup_strecher
from infinigen.assets.objects.tables.table_top import nodegroup_generate_table_top
from infinigen.assets.objects.tables.table_utils import (
    nodegroup_create_anchors,
    nodegroup_create_legs_and_strechers,
)
from infinigen.core import surface, tagging
from infinigen.core import tags as t
from infinigen.core.nodes import node_utils

# from infinigen.assets.materials import metal, metal_shader_list
# from infinigen.assets.materials.fabrics import fabric
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import (
    AssetParameters,
    ParameterizedAssetFactory,
)
from infinigen.core.surface import NoApply
from infinigen.core.util.math import FixedSeed, int_hash
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
            "Profile Aspect Ratio": kwargs["Top Profile Aspect Ratio"],
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

    elif kwargs["Leg Style"] == "square":
        leg = nw.new_node(
            nodegroup_generate_leg_square(**kwargs).name,
            input_kwargs={
                "Height": kwargs["Leg Height"],
                "Width": 0.707
                * kwargs["Leg Placement Top Relative Scale"]
                * kwargs["Top Profile Width"]
                * kwargs["Top Profile Aspect Ratio"],
                "Has Bottom Connector": (kwargs["Strecher Increament"] > 0),
                "Profile Width": kwargs["Leg Diameter"],
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

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry},
        attrs={"is_active_output": True},
    )


class TableDiningParameters(AssetParameters):
    width: Annotated[float, Field(ge=0.91, le=1.16, json_schema_extra={"editable": False})]
    dimensions: Annotated[float, Field(ge=0.65, le=0.85, json_schema_extra={"editable": True})]
    top_thickness: Annotated[
        float, Field(ge=0.03, le=0.06, json_schema_extra={"editable": False})
    ]
    strecher_relative_pos: Annotated[
        float, Field(ge=0.2, le=0.6, json_schema_extra={"editable": False})
    ]
    dining_table_230: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": False})
    ] = 1.0
    strecher_increament: Annotated[
        int,
        Field(
            json_schema_extra={"editable": False, "kind": "enum", "choices": [0, 1, 2]}
        ),
    ] = 1


class CoffeeTableParameters(TableDiningParameters):
    width: Annotated[float, Field(ge=0.6, le=0.9, json_schema_extra={"editable": False})]
    dimensions: Annotated[float, Field(ge=0.4, le=0.5, json_schema_extra={"editable": True})]
    # NOTE: effect depends on sampled leg style.
    top_thickness: Annotated[
        float, Field(ge=0.03, le=0.06, json_schema_extra={"editable": False})
    ]
    # NOTE: only applies when Leg Style is straight.
    strecher_relative_pos: Annotated[
        float, Field(ge=0.2, le=0.6, json_schema_extra={"editable": False})
    ]


class SideTableParameters(TableDiningParameters):
    width: Annotated[float, Field(ge=0.45, le=0.65, json_schema_extra={"editable": False})]
    dimensions: Annotated[float, Field(ge=0.4, le=0.65, json_schema_extra={"editable": True})]
    top_thickness: Annotated[
        float, Field(ge=0.03, le=0.06, json_schema_extra={"editable": False})
    ]
    strecher_relative_pos: Annotated[
        float, Field(ge=0.2, le=0.6, json_schema_extra={"editable": False})
    ]


class TableDiningFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = TableDiningParameters

    def __init__(self, factory_seed, coarse=False, dimensions=None):
        self._table_dimensions = dimensions
        super(TableDiningFactory, self).__init__(factory_seed, coarse=coarse)
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

    def _sample_model_field(self, field_name: str) -> float:
        field_schema = self.parameters_model.model_json_schema()["properties"][field_name]
        return uniform(field_schema["minimum"], field_schema["maximum"])

    def _resolve_table_dimensions(
        self, params: TableDiningParameters, spawn: dict[str, Any]
    ) -> tuple[float, float, float]:
        if self._table_dimensions is not None:
            return self._table_dimensions
        return spawn["length"], params.width, params.dimensions

    def _spawn_length(self, params: TableDiningParameters) -> float:
        if self._table_dimensions is not None:
            return self._table_dimensions[0]
        if params.dining_table_230 < 0.7:
            return uniform(1.4, 2.8)
        schema = self.parameters_model.model_json_schema()["properties"]["width"]
        with FixedSeed(int_hash((params.seed, "table_length"))):
            return uniform(schema["minimum"], schema["maximum"])

    def _sample_geometry_spawn_state(self, params: TableDiningParameters) -> dict[str, Any]:
        if self._table_dimensions is not None:
            x, y, _z = self._table_dimensions
            length = x
            width = y
        else:
            length = self._spawn_length(params)
            width = params.width
        leg_style = choice(["straight", "single_stand", "square"], p=[0.5, 0.1, 0.4])
        if leg_style == "single_stand":
            leg_number = 2
            leg_diameter = uniform(0.22 * length, 0.28 * length)
            leg_curve_ctrl_pts = [
                (0.0, uniform(0.1, 0.2)),
                (0.5, uniform(0.1, 0.2)),
                (0.9, uniform(0.2, 0.3)),
                (1.0, 1.0),
            ]
            top_scale = uniform(0.6, 0.7)
            bottom_scale = 1.0
        elif leg_style == "square":
            leg_number = 2
            leg_diameter = uniform(0.07, 0.10)
            leg_curve_ctrl_pts = None
            top_scale = 0.8
            bottom_scale = 1.0
        elif leg_style == "straight":
            leg_diameter = uniform(0.05, 0.07)
            leg_number = 4
            leg_curve_ctrl_pts = [
                (0.0, 1.0),
                (0.4, uniform(0.85, 0.95)),
                (1.0, uniform(0.4, 0.6)),
            ]
            top_scale = 0.8
            bottom_scale = uniform(1.0, 1.2)
        else:
            raise NotImplementedError
        return {
            "length": length,
            "width": width,
            "leg_style": leg_style,
            "leg_number": leg_number,
            "leg_diameter": leg_diameter,
            "leg_curve_ctrl_pts": leg_curve_ctrl_pts,
            "top_scale": top_scale,
            "bottom_scale": bottom_scale,
        }

    def _build_geometry_params(
        self, params: TableDiningParameters, spawn: dict[str, Any]
    ) -> dict[str, Any]:
        x, y, z = self._resolve_table_dimensions(params, spawn)
        return {
            "Top Profile N-gon": 4,
            "Top Profile Width": 1.414 * x,
            "Top Profile Aspect Ratio": y / x,
            "Top Profile Fillet Ratio": self.top_profile_fillet_ratio,
            # NOTE: top_thickness effect depends on sampled leg_style branch.
            "Top Thickness": params.top_thickness,
            "Top Vertical Fillet Ratio": self.top_vertical_fillet_ratio,
            "Height": z,
            "Top Height": z - params.top_thickness,
            "Leg Number": spawn["leg_number"],
            "Leg Style": spawn["leg_style"],
            "Leg NGon": 4,
            "Leg Placement Top Relative Scale": spawn["top_scale"],
            "Leg Placement Bottom Relative Scale": spawn["bottom_scale"],
            "Leg Height": 1.0,
            "Leg Diameter": spawn["leg_diameter"],
            "Leg Curve Control Points": spawn["leg_curve_ctrl_pts"],
            # NOTE: strecher_relative_pos only affects geometry when leg_style is straight (~50% of seeds).
            "Strecher Relative Pos": params.strecher_relative_pos,
            "Strecher Increament": params.strecher_increament,
        }

    def _sample_init_parameters(self, seed: int) -> TableDiningParameters:
        material_params, scratch, edge_wear = self._sample_materials(seed)
        self._material_params = material_params
        self._scratch = scratch
        self._edge_wear = edge_wear
        if self._table_dimensions is not None:
            _x, width, height = self._table_dimensions
        else:
            width = self._sample_model_field("width")
            height = self._sample_model_field("dimensions")
        return self.parameters_model(
            seed=seed,
            width=width,
            dimensions=height,
            top_thickness=uniform(0.03, 0.06),
            strecher_relative_pos=uniform(0.2, 0.6),
            dining_table_230=1.0,
            strecher_increament=1,
        )

    def _sample_spawn_parameters(
        self, params: TableDiningParameters, seed: int, i: int
    ) -> TableDiningParameters:
        self._geometry_spawn = self._sample_geometry_spawn_state(params)
        return params

    def apply_parameters(
        self, params: TableDiningParameters, *, spawn_scope: bool = True
    ) -> None:
        if not hasattr(self, "_material_params"):
            material_params, scratch, edge_wear = self._sample_materials(params.seed)
            self._material_params = material_params
            self._scratch = scratch
            self._edge_wear = edge_wear
        # NOTE: top_vertical_fillet_ratio and top_profile_fillet_ratio do not elicit a reliable visual change in exported geometry; sampled on self from seed, excluded from quartet sampling.
        with FixedSeed(params.seed):
            self.top_vertical_fillet_ratio = uniform(0.1, 0.3)
            self.top_profile_fillet_ratio = uniform(0.0, 0.02)
        # NOTE: top_thickness and strecher_relative_pos effects vary by leg_style spawn branch (stretchers only on straight legs); excluded from quartet sampling.
        self.dimensions = self._table_dimensions
        if spawn_scope and hasattr(self, "_geometry_spawn"):
            spawn = self._geometry_spawn
        else:
            spawn = self._sample_geometry_spawn_state(params)
        self.params = {
            **self._build_geometry_params(params, spawn),
            **self._material_params,
        }
        self.clothes_scatter = NoApply()
        self.scratch = self._scratch
        self.edge_wear = self._edge_wear
        self._use_fixed_spawn_draws = spawn_scope

    def create_asset(self, **params):
        bpy.ops.mesh.primitive_plane_add(
            size=2,
            enter_editmode=False,
            align="WORLD",
            location=(0, 0, 0),
            scale=(1, 1, 1),
        )
        obj = bpy.context.active_object

        # surface.add_geomod(obj, geometry_assemble_table, apply=False, input_kwargs=self.params)
        surface.add_geomod(
            obj, geometry_assemble_table, apply=True, input_kwargs=self.params
        )
        tagging.tag_system.relabel_obj(obj)
        assert tagging.tagged_face_mask(obj, {t.Subpart.SupportSurface}).sum() != 0

        return obj

    def finalize_assets(self, assets):
        pass
        # if self.scratch:
        #     self.scratch.apply(assets)
        # if self.edge_wear:
        #     self.edge_wear.apply(assets)

    # def finalize_assets(self, assets):
    #    self.clothes_scatter.apply(assets)


class SideTableFactory(TableDiningFactory):
    parameters_model: ClassVar[type[AssetParameters]] = SideTableParameters

    def __init__(self, factory_seed, coarse=False, dimensions=None):
        super().__init__(factory_seed, coarse=coarse, dimensions=dimensions)

    def apply_parameters(
        self, params: SideTableParameters, *, spawn_scope: bool = True
    ) -> None:
        # NOTE: top_profile_fillet_ratio and top_vertical_fillet_ratio do not elicit a clear visual change in exported geometry; excluded from quartet sampling.
        with FixedSeed(params.seed):
            self.top_profile_fillet_ratio = uniform(0.0, 0.02)
            self.top_vertical_fillet_ratio = uniform(0.1, 0.3)
        super().apply_parameters(params, spawn_scope=spawn_scope)

    def _build_geometry_params(
        self, params: SideTableParameters, spawn: dict[str, Any]
    ) -> dict[str, Any]:
        extended = TableDiningParameters.model_construct(
            **params.model_dump(),
            top_profile_fillet_ratio=self.top_profile_fillet_ratio,
            top_vertical_fillet_ratio=self.top_vertical_fillet_ratio,
        )
        return TableDiningFactory._build_geometry_params(self, extended, spawn)


class CoffeeTableFactory(TableDiningFactory):
    parameters_model: ClassVar[type[AssetParameters]] = CoffeeTableParameters

    def __init__(self, factory_seed, coarse=False, dimensions=None):
        super().__init__(factory_seed, coarse=coarse, dimensions=dimensions)

    def _sample_init_parameters(self, seed: int) -> CoffeeTableParameters:
        material_params, scratch, edge_wear = self._sample_materials(seed)
        self._material_params = material_params
        self._scratch = scratch
        self._edge_wear = edge_wear
        if self._table_dimensions is not None:
            _x, width, height = self._table_dimensions
        else:
            width = self._sample_model_field("width")
            height = self._sample_model_field("dimensions")
        return CoffeeTableParameters(
            seed=seed,
            width=width,
            dimensions=height,
            top_thickness=uniform(0.03, 0.06),
            strecher_relative_pos=uniform(0.2, 0.6),
            dining_table_230=1.0,
            strecher_increament=1,
        )

    def apply_parameters(
        self, params: CoffeeTableParameters, *, spawn_scope: bool = True
    ) -> None:
        # NOTE: top_profile_fillet_ratio and top_vertical_fillet_ratio do not elicit a clear visual change in exported geometry; excluded from quartet sampling.
        with FixedSeed(params.seed):
            self.top_profile_fillet_ratio = uniform(0.0, 0.02)
            self.top_vertical_fillet_ratio = uniform(0.1, 0.3)
        super().apply_parameters(params, spawn_scope=spawn_scope)

    def _build_geometry_params(
        self, params: TableDiningParameters, spawn: dict[str, Any]
    ) -> dict[str, Any]:
        geometry = super()._build_geometry_params(params, spawn)
        geometry["Top Profile Fillet Ratio"] = self.top_profile_fillet_ratio
        geometry["Top Vertical Fillet Ratio"] = self.top_vertical_fillet_ratio
        return geometry
