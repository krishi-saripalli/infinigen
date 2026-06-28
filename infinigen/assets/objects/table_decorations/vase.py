from __future__ import annotations

# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.
# Authors: Yiming Zuo
from typing import Annotated, ClassVar

import bpy
from numpy.random import choice, randint, uniform
from pydantic import Field

import infinigen.core.util.blender as butil
from infinigen.assets.composition import material_assignments
from infinigen.assets.objects.table_decorations.utils import (
    nodegroup_lofting,
    nodegroup_star_profile,
)
from infinigen.core import surface
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import (
    AssetParameters,
    ParameterizedAssetFactory,
)
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import weighted_sample


class VaseFactoryParameters(AssetParameters):
    height: Annotated[float, Field(ge=0.17, le=0.5, json_schema_extra={"editable": True})]
    diameter: Annotated[float, Field(ge=0.05, le=0.3, json_schema_extra={"editable": True})]
    top_scale: Annotated[float, Field(ge=0.16, le=0.96, json_schema_extra={"editable": False})]
    neck_position: Annotated[
        float, Field(ge=0.55, le=0.95, json_schema_extra={"editable": True})
    ]
    neck_scale: Annotated[float, Field(ge=0.2, le=0.8, json_schema_extra={"editable": True})]
    shoulder_position: Annotated[
        float, Field(ge=0.3, le=0.7, json_schema_extra={"editable": True})
    ]
    shoulder_thickness: Annotated[
        float, Field(ge=0.1, le=0.25, json_schema_extra={"editable": False})
    ]
    foot_scale: Annotated[float, Field(ge=0.4, le=0.6, json_schema_extra={"editable": False})]
    foot_height: Annotated[float, Field(ge=0.01, le=0.1, json_schema_extra={"editable": False})]



class VaseFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = VaseFactoryParameters

    def __init__(self, factory_seed, coarse=False, dimensions=None):
        self._dimensions = dimensions
        super(VaseFactory, self).__init__(factory_seed, coarse=coarse)
        self.init_legacy_parameters()

    def _sample_material_state(self, seed: int) -> None:
        with FixedSeed(seed):
            params = {
                "Material": weighted_sample(
                    material_assignments.marble + material_assignments.tableware
                )(),
            }
            self._material_params = {k: v() for k, v in params.items()}
            scratch_prob, edge_wear_prob = material_assignments.wear_tear_prob
            scratch, edge_wear = material_assignments.wear_tear
            self._scratch = None if uniform() > scratch_prob else scratch()
            self._edge_wear = None if uniform() > edge_wear_prob else edge_wear()

    def _sample_init_parameters(self, seed: int) -> VaseFactoryParameters:
        if self._dimensions is None:
            z = uniform(0.17, 0.5)
            x = z * uniform(0.3, 0.6)
        else:
            x, _, z = self._dimensions
        neck_scale = uniform(0.2, 0.8)
        self._sample_material_state(seed)
        return VaseFactoryParameters(
            seed=seed,
            height=z,
            diameter=x,
            top_scale=neck_scale * uniform(0.8, 1.2),
            neck_position=0.5 * neck_scale + 0.5 + uniform(-0.05, 0.05),
            neck_scale=neck_scale,
            shoulder_position=uniform(0.3, 0.7),
            shoulder_thickness=uniform(0.1, 0.25),
            foot_scale=uniform(0.4, 0.6),
            foot_height=uniform(0.01, 0.1),
        )

    def _sample_spawn_parameters(
        self, params: VaseFactoryParameters, seed: int, i: int
    ) -> VaseFactoryParameters:
        return params

    def apply_parameters(
        self, params: VaseFactoryParameters, *, spawn_scope: bool = True
    ) -> None:
        self._sample_material_state(params.seed)
        x = params.diameter
        z = params.height
        self.dimensions = (x, x, z)
        # NOTE: neck_mid_position and profile_star_points do not elicit a clear visual change in exported geometry; excluded from quartet sampling.
        with FixedSeed(params.seed):
            self.neck_mid_position = uniform(0.7, 0.95)
            self.profile_star_points = int(randint(16, 33))
        # NOTE: top_scale, shoulder_thickness, foot_scale, and foot_height effects vary with neck/foot profile branches; excluded from quartet sampling.
        self.params = {
            "Profile Inner Radius": 1.0,
            "Profile Star Points": self.profile_star_points,
            "U_resolution": 64,
            "V_resolution": 64,
            "Height": z,
            "Diameter": x,
            "Top Scale": params.top_scale,
            "Neck Mid Position": self.neck_mid_position,
            "Neck Position": params.neck_position,
            "Neck Scale": params.neck_scale,
            "Shoulder Position": params.shoulder_position,
            "Shoulder Thickness": params.shoulder_thickness,
            "Foot Scale": params.foot_scale,
            "Foot Height": params.foot_height,
            **self._material_params,
        }
        self.scratch = self._scratch
        self.edge_wear = self._edge_wear
        self._use_fixed_spawn_draws = spawn_scope

    def get_material_params(self):
        params = {
            "Material": weighted_sample(
                material_assignments.marble + material_assignments.tableware
            )(),
        }
        wrapped_params = {k: v() for k, v in params.items()}

        scratch_prob, edge_wear_prob = material_assignments.wear_tear_prob
        scratch, edge_wear = material_assignments.wear_tear
        scratch = None if uniform() > scratch_prob else scratch()
        edge_wear = None if uniform() > edge_wear_prob else edge_wear()

        return wrapped_params, scratch, edge_wear

    @staticmethod
    def sample_parameters(dimensions):
        # all in meters
        if dimensions is None:
            z = uniform(0.25, 0.40)
            x = uniform(0.2, 0.4) * z
            dimensions = (x, x, z)

        x, y, z = dimensions

        U_resolution = 64
        V_resolution = 64

        neck_scale = uniform(0.2, 0.8)

        parameters = {
            "Profile Inner Radius": choice([1.0, uniform(0.8, 1.0)]),
            "Profile Star Points": randint(16, U_resolution // 2 + 1),
            "U_resolution": U_resolution,
            "V_resolution": V_resolution,
            "Height": z,
            "Diameter": x,
            "Top Scale": neck_scale * uniform(0.8, 1.2),
            "Neck Mid Position": uniform(0.7, 0.95),
            "Neck Position": 0.5 * neck_scale + 0.5 + uniform(-0.05, 0.05),
            "Neck Scale": neck_scale,
            "Shoulder Position": uniform(0.3, 0.7),
            "Shoulder Thickness": uniform(0.1, 0.25),
            "Foot Scale": uniform(0.4, 0.6),
            "Foot Height": uniform(0.01, 0.1),
            "Material": choice(["glass", "ceramic"]),
        }

        return parameters

    def create_asset(self, **params):
        bpy.ops.mesh.primitive_plane_add(
            size=2,
            enter_editmode=False,
            align="WORLD",
            location=(0, 0, 0),
            scale=(1, 1, 1),
        )
        obj = bpy.context.active_object

        surface.add_geomod(obj, geometry_vases, apply=True, input_kwargs=self.params)
        butil.modify_mesh(obj, "SOLIDIFY", apply=True, thickness=0.002)
        butil.modify_mesh(obj, "SUBSURF", apply=True, levels=2, render_levels=2)

        return obj

    def finalize_assets(self, assets):
        if self.scratch:
            self.scratch.apply(assets)
        if self.edge_wear:
            self.edge_wear.apply(assets)


@node_utils.to_nodegroup(
    "nodegroup_vase_profile", singleton=False, type="GeometryNodeTree"
)
def nodegroup_vase_profile(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Profile Curve", None),
            ("NodeSocketFloat", "Height", 0.0000),
            ("NodeSocketFloat", "Diameter", 0.0000),
            ("NodeSocketFloat", "Top Scale", 0.0000),
            ("NodeSocketFloat", "Neck Mid Position", 0.0000),
            ("NodeSocketFloat", "Neck Position", 0.5000),
            ("NodeSocketFloat", "Neck Scale", 0.0000),
            ("NodeSocketFloat", "Shoulder Position", 0.0000),
            ("NodeSocketFloat", "Shoulder Thickness", 0.0000),
            ("NodeSocketFloat", "Foot Scale", 0.0000),
            ("NodeSocketFloat", "Foot Height", 0.0000),
        ],
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": group_input.outputs["Height"]}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Top Scale"],
            1: group_input.outputs["Diameter"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    neck_top = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": group_input.outputs["Profile Curve"],
            "Translation": combine_xyz_1,
            "Scale": multiply,
        },
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Height"],
            1: group_input.outputs["Neck Position"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_1})

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Diameter"],
            1: group_input.outputs["Neck Scale"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    neck = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": group_input.outputs["Profile Curve"],
            "Translation": combine_xyz,
            "Scale": multiply_2,
        },
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: 1.0000, 1: group_input.outputs["Neck Position"]},
        attrs={"use_clamp": True, "operation": "SUBTRACT"},
    )

    multiply_add = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: subtract,
            1: group_input.outputs["Neck Mid Position"],
            2: group_input.outputs["Neck Position"],
        },
        attrs={"operation": "MULTIPLY_ADD"},
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_add, 1: group_input.outputs["Height"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_3})

    add = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Neck Scale"],
            1: group_input.outputs["Top Scale"],
        },
    )

    divide = nw.new_node(
        Nodes.Math, input_kwargs={0: add, 1: 2.0000}, attrs={"operation": "DIVIDE"}
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Diameter"], 1: divide},
        attrs={"operation": "MULTIPLY"},
    )

    neck_middle = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": group_input.outputs["Profile Curve"],
            "Translation": combine_xyz_2,
            "Scale": multiply_4,
        },
    )

    neck_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [neck, neck_middle, neck_top]}
    )

    map_range = nw.new_node(
        Nodes.MapRange,
        input_kwargs={
            "Value": group_input.outputs["Shoulder Position"],
            3: group_input.outputs["Foot Height"],
            4: group_input.outputs["Neck Position"],
        },
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Neck Position"],
            1: group_input.outputs["Foot Height"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    multiply_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_1, 1: group_input.outputs["Shoulder Thickness"]},
        attrs={"operation": "MULTIPLY"},
    )

    add_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: map_range.outputs["Result"], 1: multiply_5}
    )

    minimum = nw.new_node(
        Nodes.Math,
        input_kwargs={0: add_1, 1: group_input.outputs["Neck Position"]},
        attrs={"operation": "MINIMUM"},
    )

    multiply_6 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: minimum, 1: group_input.outputs["Height"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_6})

    body_top = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": group_input.outputs["Profile Curve"],
            "Translation": combine_xyz_3,
            "Scale": group_input.outputs["Diameter"],
        },
    )

    subtract_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: map_range.outputs["Result"], 1: multiply_5},
        attrs={"operation": "SUBTRACT"},
    )

    maximum = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_2, 1: group_input.outputs["Foot Height"]},
        attrs={"operation": "MAXIMUM"},
    )

    multiply_7 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: maximum, 1: group_input.outputs["Height"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_5 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_7})

    body_bottom = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": group_input.outputs["Profile Curve"],
            "Translation": combine_xyz_5,
            "Scale": group_input.outputs["Diameter"],
        },
    )

    body_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [body_bottom, body_top]}
    )

    multiply_8 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Foot Height"],
            1: group_input.outputs["Height"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_8})

    multiply_9 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Diameter"],
            1: group_input.outputs["Foot Scale"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    foot_top = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": group_input,
            "Translation": combine_xyz_4,
            "Scale": multiply_9,
        },
    )

    foot_bottom = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": group_input, "Scale": multiply_9}
    )

    foot_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [foot_bottom, foot_top]}
    )

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [foot_geometry, body_geometry, neck_geometry]},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry_2},
        attrs={"is_active_output": True},
    )


def geometry_vases(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.6.4 of the node_transpiler
    starprofile = nw.new_node(
        nodegroup_star_profile().name,
        input_kwargs={
            "Resolution": kwargs["U_resolution"],
            "Points": kwargs["Profile Star Points"],
            "Inner Radius": kwargs["Profile Inner Radius"],
        },
    )

    vaseprofile = nw.new_node(
        nodegroup_vase_profile().name,
        input_kwargs={
            "Profile Curve": starprofile.outputs["Curve"],
            "Height": kwargs["Height"],
            "Diameter": kwargs["Diameter"],
            "Top Scale": kwargs["Top Scale"],
            "Neck Mid Position": kwargs["Neck Mid Position"],
            "Neck Position": kwargs["Neck Position"],
            "Neck Scale": kwargs["Neck Scale"],
            "Shoulder Position": kwargs["Shoulder Position"],
            "Shoulder Thickness": kwargs["Shoulder Thickness"],
            "Foot Scale": kwargs["Foot Scale"],
            "Foot Height": kwargs["Foot Height"],
        },
    )

    lofting = nw.new_node(
        nodegroup_lofting().name,
        input_kwargs={
            "Profile Curves": vaseprofile,
            "U Resolution": 64,
            "V Resolution": 64,
        },
    )

    delete_geometry = nw.new_node(
        Nodes.DeleteGeometry,
        input_kwargs={
            "Geometry": lofting.outputs["Geometry"],
            "Selection": lofting.outputs["Top"],
        },
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": delete_geometry, "Material": kwargs["Material"]},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": set_material},
        attrs={"is_active_output": True},
    )
