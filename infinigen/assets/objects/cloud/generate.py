# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Hei Law

from __future__ import annotations

from typing import Annotated, Any, ClassVar

import bpy
import gin
import numpy as np
from numpy.random import uniform
from pydantic import Field

from infinigen.assets.utils.object import new_cube
from infinigen.core import surface
from infinigen.core.nodes.node_wrangler import Nodes
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import (
    AssetParameters,
    ParameterizedAssetFactory,
)
from infinigen.core.tagging import tag_object
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import random_general as rg

from .cloud import (
    Altocumulus,
    Cumulonimbus,
    Cumulus,
    Stratocumulus,
    create_3d_grid,
)


class CloudParameters(AssetParameters):
    cloud_type_draw: Annotated[
        float, Field(ge=0.0, le=1.0, json_schema_extra={"editable": False})
    ] = 0.0
    first_pt_x: Annotated[
        float, Field(ge=-0.95, le=-0.5, json_schema_extra={"editable": False})
    ] = -0.875
    forth_pt_y: Annotated[
        float, Field(ge=-1.0, le=1.0, json_schema_extra={"editable": False})
    ] = 0.95
    mix_factor: Annotated[
        float, Field(ge=0.3, le=0.8, json_schema_extra={"editable": False})
    ] = 0.55
    noise_detail: Annotated[
        float, Field(ge=1.0, le=16.0, json_schema_extra={"editable": False})
    ] = 8.5
    noise_scale: Annotated[
        float, Field(ge=8.0, le=16.0, json_schema_extra={"editable": False})
    ] = 12.0
    rotate_angle: Annotated[
        float, Field(ge=0.0, le=0.785398, json_schema_extra={"editable": False})
    ] = 0.0
    # scale_x/y/z are the cloud's overall 3-axis size + noise scale. Their
    # schema bounds span the whole cloud family (cumulus scale_z ~16-32,
    # cumulonimbus up to 512), so for any single subtype the sweep snaps
    # scale to a value ~16x out of its own distribution, collapsing the
    # cloud into a degenerate thin sliver. They're also largely removed by
    # bbox normalization. The lobe-profile params below (y_lobe_*) are the
    # real, in-distribution shape DOFs, so restrict sampling to those.
    scale_x: Annotated[
        float, Field(ge=28.818331, le=1024.0, json_schema_extra={"editable": False})
    ] = 46.0
    scale_y: Annotated[
        float, Field(ge=0.5, le=2048.0, json_schema_extra={"editable": False})
    ] = 1.0
    scale_z: Annotated[
        float, Field(ge=16.0, le=512.0, json_schema_extra={"editable": False})
    ] = 24.0
    second_pt_y: Annotated[
        float, Field(ge=0.5, le=0.85, json_schema_extra={"editable": False})
    ] = 0.825
    third_pt_x: Annotated[
        float, Field(ge=0.25, le=0.75, json_schema_extra={"editable": False})
    ] = 0.5
    third_pt_y: Annotated[
        float, Field(ge=0.75, le=0.95, json_schema_extra={"editable": False})
    ] = 0.825
    voronoi_scale: Annotated[
        float, Field(ge=2.0, le=6.0, json_schema_extra={"editable": False})
    ] = 4.0
    scatter_voronoi_scale: Annotated[
        float, Field(ge=1.0, le=4.0, json_schema_extra={"editable": False})
    ] = 2.5
    scatter_vertices_x: Annotated[
        int, Field(ge=4, le=12, json_schema_extra={"editable": False})
    ] = 8
    scatter_vertices_y: Annotated[
        int, Field(ge=4, le=12, json_schema_extra={"editable": False})
    ] = 8
    y_lobe_peak: Annotated[
        float, Field(ge=0.8, le=1.0, json_schema_extra={"editable": True})
    ] = 0.9
    y_lobe_trough: Annotated[
        float, Field(ge=-1.0, le=-0.8, json_schema_extra={"editable": True})
    ] = -0.9
    y_lobe_count: Annotated[
        int, Field(ge=2, le=5, json_schema_extra={"editable": True})
    ] = 3


class CumulusParameters(CloudParameters):
    pass


def _z_curve_pts(params: CloudParameters, first_y: float = -1.0) -> list[list[float]]:
    return [
        [params.first_pt_x, first_y],
        [0.0, params.second_pt_y],
        [params.third_pt_x, params.third_pt_y],
        [1.0, params.forth_pt_y],
    ]


def _geo_params_from_cloud(
    params: CloudParameters, emission: float = 0.0, *, anisotropy: float
) -> dict:
    return {
        "density": 1.0,
        "anisotropy": anisotropy,
        "noise_scale": params.noise_scale,
        "noise_detail": params.noise_detail,
        "voronoi_scale": params.voronoi_scale,
        "mix_factor": params.mix_factor,
        "rotate_angle": params.rotate_angle,
        "emission_strength": emission,
        "scale": [params.scale_x, params.scale_y, params.scale_z],
    }


def _cloud_resolution(factory: Any, cloud_type: type, distance: float) -> int:
    resolution_min, resolution_max = factory.resolutions[cloud_type]
    resolution = max(1 - distance / factory.max_distance, 0)
    resolution = resolution * (resolution_max - resolution_min) + resolution_min
    return int(resolution)


def _sample_cumulus_spawn(params: CloudParameters) -> CloudParameters:
    scale_z = uniform(16.0, 32.0)
    return params.model_copy(
        update={
            "first_pt_x": uniform(-0.95, -0.8),
            "forth_pt_y": uniform(0.9, 1.0),
            "mix_factor": uniform(0.3, 0.8),
            "noise_detail": uniform(1.0, 16.0),
            "noise_scale": uniform(8.0, 16.0),
            "rotate_angle": uniform(0.0, np.pi / 4),
            "scale_z": scale_z,
            "scale_x": uniform(28.818331, 63.379149),
            "scale_y": uniform(0.5, 2.0),
            "second_pt_y": uniform(0.8, 0.85),
            "third_pt_x": uniform(0.25, 0.75),
            "third_pt_y": uniform(0.75, 0.9),
            "voronoi_scale": uniform(2.0, 6.0),
        }
    )


def _sample_cumulonimbus_spawn(params: CloudParameters) -> CloudParameters:
    scale_x = uniform(512.0, 1024.0)
    return params.model_copy(
        update={
            "first_pt_x": uniform(-0.65, -0.5),
            "forth_pt_y": uniform(-1.0, 0.5),
            "mix_factor": uniform(0.3, 0.8),
            "noise_detail": uniform(1.0, 16.0),
            "noise_scale": uniform(8.0, 16.0),
            "rotate_angle": uniform(0.0, np.pi / 4),
            "scale_x": scale_x,
            "scale_y": uniform(0.5, 2.0) * scale_x,
            "scale_z": uniform(256.0, 512.0),
            "second_pt_y": uniform(0.5, 0.7),
            "third_pt_x": uniform(0.25, 0.75),
            "third_pt_y": uniform(0.8, 0.95),
            "voronoi_scale": uniform(2.0, 6.0),
        }
    )


def _sample_stratocumulus_spawn(params: CloudParameters) -> CloudParameters:
    scale_x = uniform(128.0, 256.0)
    return params.model_copy(
        update={
            "first_pt_x": uniform(-0.95, -0.8),
            "forth_pt_y": uniform(0.9, 1.0),
            "mix_factor": uniform(0.3, 0.8),
            "noise_detail": uniform(1.0, 16.0),
            "noise_scale": uniform(8.0, 16.0),
            "rotate_angle": uniform(0.0, np.pi / 4),
            "scale_x": scale_x,
            "scale_y": uniform(0.5, 2.0) * scale_x,
            "scale_z": uniform(16.0, 32.0),
            "second_pt_y": uniform(0.8, 0.85),
            "third_pt_x": uniform(0.25, 0.75),
            "third_pt_y": uniform(0.75, 0.9),
            "voronoi_scale": uniform(2.0, 6.0),
            "y_lobe_peak": uniform(0.8, 1.0),
            "y_lobe_trough": uniform(-1.0, -0.8),
            "y_lobe_count": int(np.random.randint(2, 6)),
        }
    )


def _sample_altocumulus_spawn(params: CloudParameters) -> CloudParameters:
    scale_z = uniform(16.0, 32.0)
    scale_x = uniform(scale_z * 1.2, scale_z * 2.0) * 4.0
    return params.model_copy(
        update={
            "first_pt_x": uniform(-0.95, -0.8),
            "forth_pt_y": uniform(0.9, 1.0),
            "mix_factor": uniform(0.3, 0.8),
            "noise_detail": uniform(1.0, 16.0),
            "noise_scale": uniform(8.0, 16.0),
            "rotate_angle": uniform(0.0, np.pi / 4),
            "scale_x": scale_x,
            "scale_y": uniform(0.5, 2.0) * scale_x,
            "scale_z": scale_z * 4.0,
            "second_pt_y": uniform(0.8, 0.85),
            "third_pt_x": uniform(0.25, 0.75),
            "third_pt_y": uniform(0.75, 0.9),
            "voronoi_scale": uniform(2.0, 6.0),
            "scatter_voronoi_scale": uniform(1.0, 4.0),
            "scatter_vertices_x": int(np.random.randint(4, 13)),
            "scatter_vertices_y": int(np.random.randint(4, 13)),
        }
    )


@gin.configurable
class CloudFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = CloudParameters

    def __init__(
        self,
        factory_seed,
        coarse=False,
        terrain_mesh=None,
        max_distance=300,
        steps=128,
        cloudy=("bool", 0.01),
    ):
        super(CloudFactory, self).__init__(factory_seed, coarse=coarse)
        self.max_distance = max_distance
        self._cloudy_gin = cloudy
        self.ref_cloud = bpy.data.meshes.new("ref_cloud")
        self.ref_cloud.from_pydata(create_3d_grid(steps=steps), [], [])
        self.ref_cloud.update()
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> CloudParameters:
        self._cloudy = rg(self._cloudy_gin)
        return CloudParameters(seed=seed)

    def _sample_spawn_parameters(
        self, params: CloudParameters, seed: int, i: int
    ) -> CloudParameters:
        return _sample_cumulus_spawn(params)

    def apply_parameters(
        self, params: CloudParameters, *, spawn_scope: bool = True
    ) -> None:
        if not hasattr(self, "_cloudy"):
            self._cloudy = rg(self._cloudy_gin)
        self.cloudy = self._cloudy
        self.cloud_types = (
            [Cumulonimbus]
            if self.cloudy
            else [Cumulus, Stratocumulus, Altocumulus]
        )
        self.resolutions = {
            Cumulonimbus: [16, 128],
            Cumulus: [16, 128],
            Stratocumulus: [32, 256],
            Altocumulus: [16, 64],
        }
        scale_resolution = 4
        self.resolutions = {
            k: (scale_resolution * u, scale_resolution * v)
            for k, (u, v) in self.resolutions.items()
        }
        self.min_distance = 256 if self.cloudy else 64
        self.dome_radius = 1024 if self.cloudy else 256
        self.dome_threshold = 32 if self.cloudy else 0
        self.density_range = [1e-5, 1e-4] if self.cloudy else [1e-4, 2e-4]
        self.max_scale = max(t.MAX_EXPECTED_SCALE for t in self.cloud_types)
        self.density = max(t.PLACEHOLDER_DENSITY for t in self.cloud_types)
        self._use_fixed_spawn_draws = spawn_scope
        with FixedSeed(params.seed):
            self._anisotropy = uniform(-0.5, 0.5)
        if spawn_scope:
            self._cloud_params = params

    def spawn_locations(self):
        obj = new_cube()
        surface.add_geomod(
            obj,
            self.geo_dome,
            apply=True,
            input_args=[
                self.dome_radius,
                self.dome_threshold,
                self.density_range,
                self.min_distance,
            ],
        )

        locations = np.array([obj.matrix_world @ v.co for v in obj.data.vertices])
        butil.delete(obj)
        return locations

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        return butil.spawn_empty("placeholder", disp_type="CUBE", s=self.max_scale)

    def create_asset(self, distance, **kwargs):
        if self._use_fixed_spawn_draws:
            p = self._cloud_params
            type_idx = min(int(p.cloud_type_draw * len(self.cloud_types)), len(self.cloud_types) - 1)
            cloud_type = self.cloud_types[type_idx]
            resolution = _cloud_resolution(self, cloud_type, distance)
            new_cloud = _make_parameterized_cloud(
                cloud_type, "Cloud", self.ref_cloud, p, resolution, self._anisotropy
            )
        else:
            cloud_type = np.random.choice(self.cloud_types)
            resolution = _cloud_resolution(self, cloud_type, distance)
            new_cloud = cloud_type("Cloud", self.ref_cloud)
            new_cloud = new_cloud.make_cloud(
                marching_cubes=False,
                resolution=resolution,
            )
        butil.apply_transform(new_cloud)
        tag_object(new_cloud, "cloud")
        return new_cloud

    @staticmethod
    def geo_dome(
        nw,
        dome_radius,
        dome_threshold,
        density_range,
        min_distance,
    ):
        ico_sphere = nw.new_node(
            "GeometryNodeMeshIcoSphere",
            input_kwargs={
                "Radius": dome_radius,
                "Subdivisions": 8,
            },
        )

        transform = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": ico_sphere,
                "Scale": (1.2, 1.4, 1.0),
            },
        )

        position = nw.new_node(Nodes.InputPosition)
        separate_xyz = nw.new_node(
            Nodes.SeparateXYZ,
            input_kwargs={
                "Vector": position,
            },
        )

        less_than = nw.new_node(
            Nodes.Math,
            input_kwargs={
                0: separate_xyz.outputs["Z"],
                1: dome_threshold,
            },
            attrs={
                "operation": "LESS_THAN",
            },
        )

        delete_geometry = nw.new_node(
            "GeometryNodeDeleteGeometry",
            input_kwargs={
                "Geometry": transform,
                "Selection": less_than,
            },
        )

        distribute_points_on_faces = nw.new_node(
            Nodes.DistributePointsOnFaces,
            input_kwargs={
                "Mesh": delete_geometry,
                "Distance Min": min_distance,
                "Density Max": np.random.uniform(*density_range),
                "Seed": np.random.randint(1e5),
            },
            attrs={
                "distribute_method": "POISSON",
            },
        )

        combine_xyz = nw.new_node(
            Nodes.CombineXYZ,
            input_kwargs={
                "Z": nw.uniform(32, np.random.randint(64, 1e5)),
            },
        )

        set_position = nw.new_node(
            Nodes.SetPosition,
            input_kwargs={
                "Geometry": distribute_points_on_faces.outputs["Points"],
                "Offset": combine_xyz,
            },
        )

        verts = nw.new_node(
            Nodes.PointsToVertices,
            input_kwargs={
                "Points": set_position,
            },
        )

        nw.new_node(
            Nodes.GroupOutput,
            input_kwargs={
                "Geometry": verts,
            },
        )


class CumulonimbusFactory(CloudFactory):
    def __init__(
        self,
        factory_seed,
        coarse=False,
        max_distance=300,
        steps=128,
    ):
        self.cloud_types = [Cumulonimbus]
        super(CumulonimbusFactory, self).__init__(
            factory_seed, coarse, max_distance, steps
        )
        self.cloud_types = [Cumulonimbus]

    def _sample_spawn_parameters(
        self, params: CloudParameters, seed: int, i: int
    ) -> CloudParameters:
        return _sample_cumulonimbus_spawn(params)

    def create_asset(self, distance, **kwargs):
        cloud_type = Cumulonimbus
        resolution = _cloud_resolution(self, cloud_type, distance)
        if self._use_fixed_spawn_draws:
            new_cloud = _make_parameterized_cloud(
                cloud_type, "Cloud", self.ref_cloud, self._cloud_params, resolution * 2, self._anisotropy
            )
        else:
            new_cloud = cloud_type("Cloud", self.ref_cloud)
            new_cloud = new_cloud.make_cloud(
                marching_cubes=False,
                resolution=resolution * 2,
            )
        butil.apply_transform(new_cloud)
        tag_object(new_cloud, "cloud")
        return new_cloud


class CumulusFactory(CloudFactory):
    parameters_model: ClassVar[type[AssetParameters]] = CumulusParameters

    def __init__(
        self,
        factory_seed,
        coarse=False,
        max_distance=300,
        steps=128,
    ):
        self.cloud_types = [Cumulus]
        super(CumulusFactory, self).__init__(factory_seed, coarse, max_distance, steps)
        self.cloud_types = [Cumulus]

    def _sample_init_parameters(self, seed: int) -> CumulusParameters:
        params = super()._sample_init_parameters(seed)
        return CumulusParameters(**params.model_dump())

    def _sample_spawn_parameters(
        self, params: CumulusParameters, seed: int, i: int
    ) -> CumulusParameters:
        return CumulusParameters(**_sample_cumulus_spawn(params).model_dump())

    def apply_parameters(
        self, params: CumulusParameters, *, spawn_scope: bool = True
    ) -> None:
        super().apply_parameters(params, spawn_scope=spawn_scope)
        if spawn_scope:
            self._cloud_params = params

    def create_asset(self, distance, **kwargs):
        cloud_type = Cumulus
        resolution = _cloud_resolution(self, cloud_type, distance)
        if self._use_fixed_spawn_draws:
            new_cloud = _make_parameterized_cloud(
                cloud_type, "Cloud", self.ref_cloud, self._cloud_params, resolution, self._anisotropy
            )
        else:
            new_cloud = cloud_type("Cloud", self.ref_cloud)
            new_cloud = new_cloud.make_cloud(
                marching_cubes=False,
                resolution=resolution,
            )
        butil.apply_transform(new_cloud)
        tag_object(new_cloud, "cloud")
        return new_cloud


class _ParameterizedCumulus(Cumulus):
    def __init__(
        self,
        name,
        ref_cloud,
        curve_pts=None,
        geo_params=None,
    ):
        self._curve_pts = curve_pts
        self._geo_params_override = geo_params
        super().__init__(name, ref_cloud)

    def get_scale(self):
        if self._geo_params_override is not None:
            return self._geo_params_override["scale"]
        return super().get_scale()

    def get_params(self):
        if self._geo_params_override is not None:
            params = dict(self._geo_params_override)
            params.pop("scale")
            return params
        return super().get_params()

    def sample_curves(self):
        if self._curve_pts is not None:
            return self._curve_pts
        return super().sample_curves()


class _ParameterizedStratocumulus(Stratocumulus):
    def __init__(
        self,
        name: str,
        ref_cloud,
        z_curve_pts: list[list[float]],
        y_lobe_count: int,
        y_lobe_peak: float,
        y_lobe_trough: float,
        geo_params: dict,
    ):
        self._z_curve_pts = z_curve_pts
        self._y_lobe_count = y_lobe_count
        self._y_lobe_peak = y_lobe_peak
        self._y_lobe_trough = y_lobe_trough
        self._geo_params_override = geo_params
        super().__init__(name, ref_cloud)

    def get_scale(self):
        return self._geo_params_override["scale"]

    def get_params(self):
        params = dict(self._geo_params_override)
        params.pop("scale")
        return params

    def sample_y_curves(self):
        n = self._y_lobe_count
        num_pts = n + n - 1
        xs = np.linspace(-1, 1, num_pts + 2)
        ys: list[float] = [-1.0]
        for i in range(len(xs[1:-1])):
            if i % 2 == 0:
                ys.append(self._y_lobe_peak)
            else:
                ys.append(self._y_lobe_trough)
        ys.append(-1.0)
        return np.stack((xs, np.array(ys)), axis=1)

    def sample_curves(self):
        return [self.sample_y_curves(), self._z_curve_pts]


class _ParameterizedAltocumulus(Altocumulus):
    def __init__(
        self, name: str, ref_cloud, params: CloudParameters, anisotropy: float
    ):
        self._params = params
        self._anisotropy = anisotropy
        self._curve_pts = _z_curve_pts(params)
        self._geo_scale = [params.scale_x, params.scale_y, params.scale_z]
        super().__init__(name, ref_cloud)

    def get_scale(self):
        return self._geo_scale

    def sample_curves(self):
        return self._curve_pts

    def get_params(self):
        n = type(self).NUM_SUBCLOUDS
        p = self._params
        scatter_params = {
            "voronoi_scale": p.scatter_voronoi_scale,
            "vertices_x": p.scatter_vertices_x,
            "vertices_y": p.scatter_vertices_y,
        }
        return {
            "densities": np.full(n, 1.0),
            "anisotropies": np.full(n, self._anisotropy),
            "noise_scales": np.full(n, p.noise_scale),
            "noise_details": np.full(n, p.noise_detail),
            "voronoi_scales": np.full(n, p.voronoi_scale),
            "mix_factors": np.full(n, p.mix_factor),
            "rotate_angles": np.full(n, p.rotate_angle),
            "emission_strengths": np.full(n, 0.0),
            "scatter_params": scatter_params,
        }


def _make_parameterized_cloud(
    cloud_type: type,
    name: str,
    ref_cloud,
    params: CloudParameters,
    resolution: int,
    anisotropy: float,
):
    if cloud_type is Cumulus:
        cloud = _ParameterizedCumulus(
            name,
            ref_cloud,
            curve_pts=_z_curve_pts(params),
            geo_params=_geo_params_from_cloud(params, anisotropy=anisotropy),
        )
    elif cloud_type is Cumulonimbus:
        cloud = _ParameterizedCumulus(
            name,
            ref_cloud,
            curve_pts=_z_curve_pts(params),
            geo_params=_geo_params_from_cloud(params, emission=0.01, anisotropy=anisotropy),
        )
    elif cloud_type is Stratocumulus:
        cloud = _ParameterizedStratocumulus(
            name,
            ref_cloud,
            z_curve_pts=_z_curve_pts(params),
            y_lobe_count=params.y_lobe_count,
            y_lobe_peak=params.y_lobe_peak,
            y_lobe_trough=params.y_lobe_trough,
            geo_params=_geo_params_from_cloud(params, anisotropy=anisotropy),
        )
    elif cloud_type is Altocumulus:
        cloud = _ParameterizedAltocumulus(name, ref_cloud, params, anisotropy)
    else:
        cloud = cloud_type(name, ref_cloud)
    return cloud.make_cloud(marching_cubes=False, resolution=resolution)


class StratocumulusFactory(CloudFactory):
    def __init__(
        self,
        factory_seed,
        coarse=False,
        max_distance=300,
        steps=128,
    ):
        self.cloud_types = [Stratocumulus]
        super(StratocumulusFactory, self).__init__(
            factory_seed, coarse, max_distance, steps
        )
        self.cloud_types = [Stratocumulus]

    def _sample_spawn_parameters(
        self, params: CloudParameters, seed: int, i: int
    ) -> CloudParameters:
        return _sample_stratocumulus_spawn(params)

    def create_asset(self, distance, **kwargs):
        cloud_type = Stratocumulus
        resolution = _cloud_resolution(self, cloud_type, distance)
        if self._use_fixed_spawn_draws:
            new_cloud = _make_parameterized_cloud(
                cloud_type, "Cloud", self.ref_cloud, self._cloud_params, resolution, self._anisotropy
            )
        else:
            new_cloud = cloud_type("Cloud", self.ref_cloud)
            new_cloud = new_cloud.make_cloud(
                marching_cubes=False,
                resolution=resolution,
            )
        butil.apply_transform(new_cloud)
        tag_object(new_cloud, "cloud")
        return new_cloud


class AltocumulusFactory(CloudFactory):
    parameters_model: ClassVar[type[AssetParameters]] = CloudParameters

    def __init__(
        self,
        factory_seed,
        coarse=False,
        max_distance=300,
        steps=128,
    ):
        self.cloud_types = [Altocumulus]
        super(AltocumulusFactory, self).__init__(
            factory_seed, coarse, max_distance=max_distance, steps=steps
        )
        self.cloud_types = [Altocumulus]

    def _sample_spawn_parameters(
        self, params: CloudParameters, seed: int, i: int
    ) -> CloudParameters:
        return _sample_altocumulus_spawn(params)

    def apply_parameters(
        self, params: CloudParameters, *, spawn_scope: bool = True
    ) -> None:
        super().apply_parameters(params, spawn_scope=spawn_scope)
        self.cloud_types = [Altocumulus]

    def create_asset(self, distance, **kwargs):
        cloud_type = Altocumulus
        resolution = _cloud_resolution(self, cloud_type, distance)
        if self._use_fixed_spawn_draws:
            new_cloud = _make_parameterized_cloud(
                cloud_type, "Cloud", self.ref_cloud, self._cloud_params, resolution, self._anisotropy
            )
        else:
            new_cloud = cloud_type("Cloud", self.ref_cloud)
            new_cloud = new_cloud.make_cloud(
                marching_cubes=False,
                resolution=resolution,
            )
        butil.apply_transform(new_cloud)
        tag_object(new_cloud, "cloud")
        return new_cloud
