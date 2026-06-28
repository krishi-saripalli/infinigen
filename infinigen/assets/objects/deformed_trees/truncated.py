# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Lingjie Mei


from __future__ import annotations

from typing import Annotated, ClassVar

import numpy as np
from numpy.random import uniform
from pydantic import Field

from infinigen.assets.objects.deformed_trees.fallen import (
    FallenTreeFactory,
    FallenTreeParameters,
)
from infinigen.assets.utils.decorate import read_co
from infinigen.core import surface
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.placement.parameters import AssetParameters
from infinigen.core.tagging import tag_object
from infinigen.core.util import blender as butil


class TruncatedTreeParameters(FallenTreeParameters):
    cut_center_z: Annotated[
        float, Field(ge=0.8, le=1.5, json_schema_extra={"editable": True})
    ]
    cut_normal_x: Annotated[
        float, Field(ge=-0.4, le=0.4, json_schema_extra={"editable": True})
    ]
    noise_strength: Annotated[
        float, Field(ge=0.6, le=1.0, json_schema_extra={"editable": True})
    ]


class TruncatedTreeFactory(FallenTreeFactory):
    parameters_model: ClassVar[type[AssetParameters]] = TruncatedTreeParameters

    def _sample_init_parameters(self, seed: int) -> TruncatedTreeParameters:
        base = super()._sample_init_parameters(seed)
        return TruncatedTreeParameters(
            **base.model_copy(
                update={
                    "cut_center_z": uniform(0.8, 1.5),
                    "cut_normal_x": uniform(-0.4, 0.4),
                    "noise_strength": uniform(0.6, 1.0),
                }
            ).model_dump()
        )

    @staticmethod
    def geo_cutter(nw: NodeWrangler, strength, scale, radius, metric_fn):
        geometry = nw.new_node(
            Nodes.GroupInput, expose_input=[("NodeSocketGeometry", "Geometry", None)]
        )
        offset = nw.scalar_multiply(
            nw.new_node(
                Nodes.Clamp,
                [
                    nw.new_node(
                        Nodes.NoiseTexture,
                        input_kwargs={
                            "Vector": nw.new_node(Nodes.InputPosition),
                            "Scale": scale,
                        },
                    ),
                    0.3,
                    0.7,
                ],
            ),
            strength,
        )
        anchors = (-1, 0), (-0.5, 0), (0, 1), (0.5, 0), (1, 0)
        offset = nw.scalar_multiply(
            offset, nw.build_float_curve(surface.eval_argument(nw, metric_fn), anchors)
        )
        geometry = nw.new_node(
            Nodes.SetPosition, [geometry, None, None, nw.combine(0, 0, offset)]
        )
        nw.new_node(Nodes.GroupOutput, input_kwargs={"Geometry": geometry})

    def create_asset(self, i, distance=0, **params):
        obj = self.build_tree(i, distance, **params)
        x, y, z = read_co(obj).T
        radius = np.amax(np.sqrt(x**2 + y**2)[z < 0.1])
        self.trunk_surface.apply(obj)
        butil.apply_modifiers(obj)
        cut_center = np.array([0, 0, self.cut_center_z])
        cut_normal = np.array([self.cut_normal_x, 0, 1])
        obj = self.build_half(
            obj,
            cut_center,
            cut_normal,
            self.noise_strength,
            self.noise_scale,
            radius,
            False,
        )
        tag_object(obj, "truncated_tree")
        return obj
