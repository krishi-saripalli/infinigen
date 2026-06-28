# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

from __future__ import annotations

from collections import defaultdict
from typing import Annotated, Any, ClassVar

import gin
import mathutils
import numpy as np
from numpy.random import normal as N
from numpy.random import uniform as U
from pydantic import Field

from infinigen.assets import materials
from infinigen.assets.composition import material_assignments
from infinigen.assets.objects.creatures import parts
from infinigen.assets.objects.creatures.util import cloth_sim, creature, genome, joining
from infinigen.assets.objects.creatures.util import hair as creature_hair
from infinigen.assets.objects.creatures.util.animation import idle, run_cycle
from infinigen.assets.objects.creatures.util.creature_util import offset_center
from infinigen.assets.objects.creatures.util.genome import Joint
from infinigen.core import surface
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import (
    AssetParameters,
    ParameterizedAssetFactory,
)
from infinigen.core.util import blender as butil
from infinigen.core.util.math import clip_gaussian
from infinigen.core.util.random import weighted_sample


def herbivore_hair():
    mat_roughness = U(0.5, 0.9)

    puff = U(0.14, 0.4)
    length = clip_gaussian(0.035, 0.03, 0.01, 0.1)

    return {
        "density": 500000,
        "clump_n": np.random.randint(10, 300),
        "avoid_features_dist": 0.06,
        "grooming": {
            "Length MinMaxScale": np.array(
                (length, length * U(1.5, 4), U(15, 60)), dtype=np.float32
            ),
            "Puff MinMaxScale": np.array(
                (puff, U(0.5, 1.3), U(15, 60)), dtype=np.float32
            ),
            "Combing": U(0.5, 1),
            "Strand Random Mag": U(0, 0.003) if U() < 0.5 else 0,
            "Strand Perlin Mag": U(0, 0.006),
            "Strand Perlin Scale": U(15, 45),
            "Tuft Spread": N(0.06, 0.025),
            "Tuft Clumping": U(0.7, 0.95),
            "Root Radius": 0.0025,
            "Post Clump Noise Mag": 0.001 * N(1, 0.15),
            "Hair Length Pct Min": U(0.5, 0.9),
        },
        "material": {
            "Roughness": mat_roughness,
            "Radial Roughness": mat_roughness + N(0, 0.07),
            "Random Roughness": 0,
            "IOR": 1.55,
        },
    }


def herbivore_genome(p: HerbivoreParameters | None = None):
    temp_dict = defaultdict(
        lambda: 0.2, {"body_herbivore_giraffe": 0.02, "body_herbivore_llama": 0.1}
    )
    body = genome.part(
        parts.generic_nurbs.NurbsBody(
            prefix="body_herbivore", tags=["body"], var=1, temperature=temp_dict
        )
    )

    neck_t = 0.67
    shoulder_bounds = np.array([[-20, -20, -20], [20, 20, 20]])
    splay = (
        p.shoulder_splay if p is not None else clip_gaussian(130, 7, 90, 130)
    ) / 180
    shoulder_t = (
        p.shoulder_t if p is not None else clip_gaussian(0.1, 0.05, 0.05, 0.2)
    )
    leg_scale = p.leg_length_scale if p is not None else float(N(1, 0.1))
    params = {
        "length_rad1_rad2": np.array((1.8, 0.1, 0.05))
        * leg_scale
        * N(1, (0.1, 0.05, 0.05), 3)
    }

    leg_rest = (0, 90, 0)  # (0, 90, 0)
    foot_rest = (0, -90, 0)
    foot_fac = parts.hoof.HoofAnkle()
    claw_fac = parts.hoof.HoofClaw()
    backleg_fac = parts.leg.QuadrupedBackLeg(params=params)
    frontleg_fac = parts.leg.QuadrupedFrontLeg(params=params)

    has_long_legs = p.has_long_legs if p is not None else U() < 0.15
    if has_long_legs:
        lenscale = U(1, 1.3)
        backleg_fac.params["length_rad1_rad2"][0] *= lenscale
        frontleg_fac.params["length_rad1_rad2"][0] *= lenscale

    for side in [-1, 1]:
        # foot = genome.part(claw_fac)
        foot = genome.attach(
            genome.part(claw_fac),
            genome.part(foot_fac),
            coord=(0.7, -1, 0),
            joint=Joint(rest=(0, 90, 0)),
            rotation_basis="global",
        )
        back_leg = genome.attach(
            foot,
            genome.part(backleg_fac),
            coord=(0.95, 1, 0.2),
            joint=Joint(rest=foot_rest),
            rotation_basis="global",
        )
        genome.attach(
            back_leg,
            body,
            coord=(shoulder_t, splay, 1),
            joint=Joint(rest=leg_rest, bounds=shoulder_bounds),
            rotation_basis="global",
            side=side,
        )

    for side in [-1, 1]:
        # foot = genome.part(claw_fac)
        foot = genome.attach(
            genome.part(claw_fac),
            genome.part(foot_fac),
            coord=(0.7, 1, 0),
            joint=Joint(rest=(0, 90, 0)),
            rotation_basis="normal",
        )
        front_leg = genome.attach(
            foot,
            genome.part(frontleg_fac),
            coord=(0.95, 0, 0.5),
            joint=Joint(rest=(0, -70, 0)),
        )
        genome.attach(
            front_leg,
            body,
            coord=(neck_t - shoulder_t, splay + 0 / 180, 0.9),
            joint=Joint(rest=leg_rest),
            rotation_basis="global",
            side=side,
        )

    temp_dict = defaultdict(lambda: 0.2, {"body_herbivore_giraffe": 0.02})
    head_var = p.head_var if p is not None else 0.5
    head_fac = parts.generic_nurbs.NurbsHead(
        prefix="head_herbivore", tags=["head"], var=head_var, temperature=temp_dict
    )
    head = genome.part(head_fac)

    eye_radius = p.eye_radius if p is not None else float(N(0.035, 0.01))
    eye_t = p.eye_t if p is not None else float(U(0.34, 0.45))
    eye_fac = parts.eye.MammalEye({"Radius": eye_radius})
    splay = U(80, 140) / 180
    r = U(0.7, 0.9)
    rot = np.array([0, 0, 0])
    for side in [-1, 1]:
        eye = genome.part(eye_fac)
        genome.attach(
            eye,
            head,
            coord=(eye_t, splay, r),
            joint=Joint(rest=rot),
            rotation_basis="normal",
            side=side,
        )

    jaw = genome.part(
        parts.head.CarnivoreJaw(
            {
                "length_rad1_rad2": (0.6 * head_fac.params["length"], 0.12, 0.08),
                "Canine Length": 0,
            }
        )
    )
    genome.attach(
        jaw,
        head,
        coord=(0.25 * N(1, 0.1), 0, 0.35 * N(1, 0.1)),
        joint=Joint(rest=(0, 10 * N(1, 0.1), 0)),
    )

    has_nose = p.has_nose if p is not None else U() < 0.7
    if has_nose:
        nose = genome.part(parts.head_detail.CatNose())
        genome.attach(nose, head, coord=(0.95, 1, 0.45), joint=Joint(rest=(0, 20, 0)))

    t, splay = U(0.15, eye_t - 0.07), N(125, 15) / 180
    ear_fac = parts.head_detail.CatEar({})
    ear_fac.params["length_rad1_rad2"] *= N(1.2, 0.1, 3)
    rot = np.array([0, -10, -23]) * N(1, 0.1, 3)
    for side in [-1, 1]:
        ear = genome.part(ear_fac)
        genome.attach(
            ear,
            head,
            coord=(t, splay, 1),
            joint=Joint(rest=rot),
            rotation_basis="normal",
            side=side,
        )

    has_horns = p.has_horns if p is not None else U() < 0.7
    if has_horns:
        horn_fac = parts.horn.Horn()
        horn_fac.params["length"] *= U(0.1, 2)
        horn_fac.params["rad1"] *= U(0.07, 1.5)
        horn_fac.params["rad2"] *= U(0.07, 1.5)
        t, splay = U(0.25, t), U(splay + 20 / 180, 130 / 180)
        rot = np.array([U(-40, 0), 0, N(120, 10)])
        for side in [-1, 1]:
            horn = genome.part(horn_fac)
            genome.attach(
                horn,
                head,
                coord=(t, splay, 0.5),
                joint=Joint(rest=rot),
                rotation_basis="global",
                side=side,
            )
    elif U() < 0:
        horn_fac = parts.horn.Horn()
        horn_fac.params["length"] *= U(0.3, 1)
        horn_fac.params["rotation_x"] = 0
        horn = genome.part(horn_fac)
        genome.attach(
            horn,
            head,
            coord=(U(0.3, 0.9), 1, 0.6),
            joint=Joint(rest=(0, -90, -90)),
            rotation_basis="global",
        )

    genome.attach(head, body, coord=(0.97, 0, 0.2), joint=Joint(rest=(0, 20, 0)))

    return genome.CreatureGenome(
        parts=body,
        postprocess_params=dict(
            animation=dict(),
            hair=herbivore_hair(),
        ),
    )


class HerbivoreParameters(AssetParameters):
    has_long_legs: Annotated[
        bool, Field(json_schema_extra={"editable": True, "kind": "bool"})
    ] = False
    has_nose: Annotated[
        bool, Field(json_schema_extra={"editable": True, "kind": "bool"})
    ] = False
    has_horns: Annotated[
        bool, Field(json_schema_extra={"editable": True, "kind": "bool"})
    ] = False
    shoulder_splay: Annotated[
        float, Field(ge=90.0, le=130.0, json_schema_extra={"editable": True})
    ]
    shoulder_t: Annotated[
        float, Field(ge=0.05, le=0.2, json_schema_extra={"editable": True})
    ]
    leg_length_scale: Annotated[
        float, Field(ge=0.7, le=1.3, json_schema_extra={"editable": True})
    ]
    eye_radius: Annotated[
        float, Field(ge=0.015, le=0.055, json_schema_extra={"editable": True})
    ]
    eye_t: Annotated[
        float, Field(ge=0.34, le=0.45, json_schema_extra={"editable": True})
    ]
    head_var: Annotated[
        float, Field(ge=0.3, le=0.7, json_schema_extra={"editable": True})
    ]


@gin.configurable
class HerbivoreFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = HerbivoreParameters
    max_distance = 40

    def __init__(
        self,
        factory_seed=None,
        bvh: mathutils.bvhtree.BVHTree = None,
        coarse: bool = False,
        animation_mode: str = None,
        hair: bool = True,
        clothsim_skin: bool = False,
        **kwargs,
    ):
        self.bvh = bvh
        self.animation_mode = animation_mode
        self.hair = hair
        self.clothsim_skin = clothsim_skin
        super().__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_materials(self) -> tuple[Any, Any, Any, Any]:
        return (
            weighted_sample(material_assignments.herbivore)(),
            materials.creature.Tongue(),
            materials.creature.Bone(),
            materials.creature.Nose(),
        )

    def _sample_init_parameters(self, seed: int) -> HerbivoreParameters:
        (
            self._body_material,
            self._tongue_material,
            self._teeth_material,
            self._nose_material,
        ) = self._sample_materials()
        return HerbivoreParameters(
            seed=seed,
            has_long_legs=bool(U() < 0.15),
            has_nose=bool(U() < 0.7),
            has_horns=bool(U() < 0.7),
            shoulder_splay=float(clip_gaussian(130, 7, 90, 130)),
            shoulder_t=float(clip_gaussian(0.1, 0.05, 0.05, 0.2)),
            leg_length_scale=float(N(1, 0.1)),
            eye_radius=float(N(0.035, 0.01)),
            eye_t=float(U(0.34, 0.45)),
            head_var=0.5,
        )

    def apply_parameters(
        self, params: HerbivoreParameters, *, spawn_scope: bool = True
    ) -> None:
        if self.hair and (self.animation_mode is not None or self.clothsim_skin):
            raise NotImplementedError(
                "Dynamic hair is not yet fully working. "
                "Please disable either hair or both of animation/clothsim"
            )
        if not hasattr(self, "_body_material"):
            (
                self._body_material,
                self._tongue_material,
                self._teeth_material,
                self._nose_material,
            ) = self._sample_materials()
        self._herbivore_params = params
        self.body_material = self._body_material
        self.tongue_material = self._tongue_material
        self.teeth_material = self._teeth_material
        self.nose_material = self._nose_material
        self._use_fixed_spawn_draws = spawn_scope

    def create_placeholder(self, **kwargs):
        return butil.spawn_cube(size=4)

    def apply_materials(self, root):
        self.body_material.apply(
            joining.get_parts(root, True) + joining.get_parts(root, False, "BodyExtra")
        )
        self.body_material.apply(joining.get_parts(root, False, "Tongue"))

        # TODO move these into the individual part generators
        self.tongue_material.apply(
            joining.get_parts(root, False, "Teeth")
            + joining.get_parts(root, False, "Claws")
        )
        self.teeth_material.apply(
            joining.get_parts(root, False, "Eyeball"), shader_kwargs={"coord": "X"}
        )
        self.nose_material.apply(joining.get_parts(root, False, "Nose"))

    def create_asset(self, i, placeholder, **kwargs):
        herbivore_params = (
            self._herbivore_params if self._use_fixed_spawn_draws else None
        )
        genome = herbivore_genome(herbivore_params)
        root, parts = creature.genome_to_creature(
            genome, name=f"herbivore({self.factory_seed}, {i})"
        )
        # tag_object(root, 'herbivore')
        offset_center(root)

        dynamic = self.animation_mode is not None

        joined, extras, arma, ik_targets = joining.join_and_rig_parts(
            root,
            parts,
            genome,
            rigging=dynamic,
            postprocess_func=self.apply_materials,
            **kwargs,
        )

        butil.parent_to(root, placeholder, no_inverse=True)

        if self.hair:
            creature_hair.configure_hair(
                joined, root, genome.postprocess_params["hair"]
            )
        if dynamic:
            if self.animation_mode == "run":
                run_cycle.animate_run(root, arma, ik_targets)
            elif self.animation_mode == "idle":
                idle.snap_iks_to_floor(ik_targets, self.bvh)
                idle.idle_body_noise_drivers(ik_targets)
            else:
                raise ValueError(f"Unrecognized mode {self.animation_mode=}")
        if self.clothsim_skin:
            rigidity = surface.write_vertex_group(
                joined, cloth_sim.local_pos_rigity_mask, apply=True
            )
            cloth_sim.bake_cloth(
                joined,
                genome.postprocess_params["skin"],
                attributes=dict(vertex_group_mass=rigidity),
            )

        return root
