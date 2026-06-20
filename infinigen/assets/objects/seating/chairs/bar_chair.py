# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo

from __future__ import annotations

from typing import Annotated, Any, ClassVar

import bpy
from numpy.random import choice, uniform
from pydantic import ConfigDict, Field

from infinigen.assets.composition import material_assignments
from infinigen.assets.objects.seating.chairs.seats.round_seats import (
    generate_round_seats,
)
from infinigen.assets.objects.tables.cocktail_table import geometry_create_legs
from infinigen.core import surface, tagging
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import AssetParameters, ParameterizedAssetFactory
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import weighted_sample


def geometry_assemble_chair(nw: NodeWrangler, **kwargs):
    generateseat = nw.new_node(
        generate_round_seats(
            thickness=kwargs["Top Thickness"],
            radius=kwargs["Top Profile Width"],
            seat_material=kwargs["SeatMaterial"],
        ).name
    )

    seat_instance = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": generateseat,
            "Translation": (0.0000, 0.0000, kwargs["Top Height"]),
        },
    )

    legs = nw.new_node(geometry_create_legs(**kwargs).name)

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [seat_instance, legs]}
    )

    nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry},
        attrs={"is_active_output": True},
    )


class BarChairParameters(AssetParameters):
    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    top_profile_width: Annotated[
        float, Field(alias="Top Profile Width", json_schema_extra={"editable": True})
    ]
    top_thickness: Annotated[
        float, Field(alias="Top Thickness", json_schema_extra={"editable": True})
    ]
    height: Annotated[float, Field(alias="Height", json_schema_extra={"editable": True})]
    leg_placement_bottom_relative_scale: Annotated[
        float,
        Field(
            alias="Leg Placement Bottom Relative Scale",
            json_schema_extra={"editable": True},
        ),
    ]


class BarChairFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = BarChairParameters

    def __init__(self, factory_seed, coarse=False, dimensions=None):
        super().__init__(factory_seed, coarse=coarse)
        self.dimensions = dimensions
        self.init_legacy_parameters()

    @staticmethod
    def _sample_materials() -> tuple[dict[str, Any], Any | None, Any | None]:
        params = {
            "SeatMaterial": weighted_sample(material_assignments.large_seat_fabric)(),
            "LegMaterial": weighted_sample(material_assignments.furniture_leg)(),
        }
        wrapped_params = {k: v() for k, v in params.items()}
        scratch_prob, edge_wear_prob = material_assignments.wear_tear_prob
        scratch, edge_wear = material_assignments.wear_tear
        scratch = None if uniform() > scratch_prob else scratch()
        edge_wear = None if uniform() > edge_wear_prob else edge_wear()
        return wrapped_params, scratch, edge_wear

    def get_material_params(self, leg_style):
        return self._sample_materials()

    @staticmethod
    def sample_parameters(dimensions):
        geometry, _ = BarChairFactory._sample_geometry_parameters(dimensions)
        return geometry

    @staticmethod
    def _sample_geometry_parameters(dimensions):
        if dimensions is None:
            x = uniform(0.35, 0.45)
            z = uniform(0.7, 1)
            dimensions = (x, x, z)

        x, _, z = dimensions
        top_thickness = uniform(0.06, 0.10)
        leg_style = choice(["straight", "single_stand", "wheeled"])

        parameters = {
            "Top Profile Width": x,
            "Top Thickness": top_thickness,
            "Height": z,
            "Top Height": z - top_thickness,
            "Leg Style": leg_style,
            "Leg NGon": choice([4, 32]),
            "Leg Placement Top Relative Scale": 0.7,
            "Leg Placement Bottom Relative Scale": uniform(1.1, 1.3),
            "Leg Height": 1.0,
        }

        if leg_style == "single_stand":
            parameters.update(
                {
                    "Leg Number": 1,
                    "Leg Diameter": uniform(0.7 * x, 0.9 * x),
                    "Leg Curve Control Points": [
                        (0.0, uniform(0.1, 0.2)),
                        (0.5, uniform(0.1, 0.2)),
                        (0.9, uniform(0.2, 0.3)),
                        (1.0, 1.0),
                    ],
                }
            )
        elif leg_style == "straight":
            parameters.update(
                {
                    "Leg Number": choice([3, 4]),
                    "Leg Diameter": uniform(0.04, 0.06),
                    "Leg Curve Control Points": [
                        (0.0, 1.0),
                        (0.4, uniform(0.85, 0.95)),
                        (1.0, uniform(0.4, 0.6)),
                    ],
                    "Strecher Relative Pos": uniform(0.6, 0.9),
                    "Strecher Increament": choice([0, 1, 2]),
                }
            )
        elif leg_style == "wheeled":
            parameters.update(
                {
                    "Leg Number": 1,
                    "Leg Pole Number": choice([4, 5]),
                    "Leg Diameter": uniform(0.03, 0.05),
                    "Leg Joint Height": uniform(0.5, 0.8) * (z - top_thickness),
                    "Leg Wheel Arc Sweep Angle": uniform(120, 240),
                    "Leg Wheel Width": uniform(0.11, 0.15),
                    "Leg Wheel Rot": uniform(0, 360),
                    "Leg Pole Length": uniform(1.6, 2.0),
                }
            )
        else:
            raise NotImplementedError

        return parameters, leg_style

    def _build_geometry_params(
        self, params: BarChairParameters
    ) -> tuple[dict[str, Any], Any | None, Any | None]:
        dimensions = (
            self.dimensions
            if self.dimensions is not None
            else (params.top_profile_width, params.top_profile_width, params.height)
        )
        with FixedSeed(params.seed):
            geometry, _ = self._sample_geometry_parameters(dimensions)
        geometry["Top Profile Width"] = params.top_profile_width
        geometry["Top Thickness"] = params.top_thickness
        geometry["Height"] = params.height
        geometry["Top Height"] = params.height - params.top_thickness
        geometry["Leg Placement Bottom Relative Scale"] = (
            params.leg_placement_bottom_relative_scale
        )
        with FixedSeed(params.seed):
            materials, scratch, edge_wear = self._sample_materials()
        return {**geometry, **materials}, scratch, edge_wear

    def _sample_init_parameters(self, seed: int) -> BarChairParameters:
        geometry, _ = self._sample_geometry_parameters(self.dimensions)
        return BarChairParameters.model_validate(
            {
                "seed": seed,
                "Top Profile Width": geometry["Top Profile Width"],
                "Top Thickness": geometry["Top Thickness"],
                "Height": geometry["Height"],
                "Leg Placement Bottom Relative Scale": geometry[
                    "Leg Placement Bottom Relative Scale"
                ],
            }
        )

    def apply_parameters(
        self, params: BarChairParameters, *, spawn_scope: bool = True
    ) -> None:
        geometry, scratch, edge_wear = self._build_geometry_params(params)
        self.params = {k: v for k, v in geometry.items() if v is not None}
        self.scratch = scratch
        self.edge_wear = edge_wear
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

        surface.add_geomod(
            obj, geometry_assemble_chair, apply=True, input_kwargs=self.params
        )
        tagging.tag_system.relabel_obj(obj)

        return obj

    def finalize_assets(self, assets):
        if self.scratch:
            self.scratch.apply(assets)
        if self.edge_wear:
            self.edge_wear.apply(assets)
