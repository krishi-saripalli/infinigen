# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Hongyu Wen: primary author
# - Alexander Raistrick: snake curve following animation

from __future__ import annotations

import logging
from typing import Annotated, ClassVar

import bpy
import gin
import numpy as np
from numpy.random import normal as N
from numpy.random import uniform as U
from pydantic import Field

from infinigen.assets.composition import material_assignments
from infinigen.assets.materials.creature import (
    bone,
    eyeball,
    nose,
    tongue,
)
from infinigen.assets.objects.creatures import parts
from infinigen.assets.objects.creatures.util import animation as creature_animation
from infinigen.assets.objects.creatures.util import creature, genome, joining
from infinigen.assets.objects.creatures.util.animation import curve_slither
from infinigen.assets.objects.creatures.util.animation.run_cycle import follow_path
from infinigen.assets.objects.creatures.util.genome import Joint
from infinigen.core.placement import animation_policy
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import (
    AssetParameters,
    ParameterizedAssetFactory,
)
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed, clip_gaussian
from infinigen.core.util.random import random_general, weighted_sample


class LizardParameters(AssetParameters):
    head_scale_x: Annotated[
        float, Field(ge=0.76, le=0.84, json_schema_extra={"editable": True})
    ]
    head_scale_y: Annotated[
        float, Field(ge=0.26, le=0.34, json_schema_extra={"editable": True})
    ]
    swim_mag: Annotated[
        float, Field(ge=41.0, le=59.0, json_schema_extra={"editable": True})
    ]
    swim_freq: Annotated[
        float, Field(ge=0.7, le=1.3, json_schema_extra={"editable": True})
    ]


class FrogParameters(AssetParameters):
    head_scale_x: Annotated[
        float, Field(ge=0.76, le=0.84, json_schema_extra={"editable": True})
    ]
    head_scale_y: Annotated[
        float, Field(ge=0.36, le=0.44, json_schema_extra={"editable": True})
    ]
    speed_m_s: Annotated[
        float, Field(ge=0.3, le=0.7, json_schema_extra={"editable": True})
    ]


class SnakeParameters(AssetParameters):
    snake_length: Annotated[
        float, Field(ge=0.5, le=3.0, json_schema_extra={"editable": True})
    ]
    width_mod: Annotated[
        float, Field(ge=0.85, le=1.15, json_schema_extra={"editable": True})
    ]
    height_mod: Annotated[
        float, Field(ge=0.85, le=1.15, json_schema_extra={"editable": True})
    ]
    head_scale_x: Annotated[
        float, Field(ge=0.76, le=0.84, json_schema_extra={"editable": True})
    ]
    eye_radius: Annotated[
        float, Field(ge=0.02, le=0.04, json_schema_extra={"editable": True})
    ]
    swim_mag: Annotated[
        float, Field(ge=170.0, le=230.0, json_schema_extra={"editable": True})
    ]


class ChameleonParameters(AssetParameters):
    body_scale: Annotated[
        float, Field(ge=0.7, le=1.3, json_schema_extra={"editable": True})
    ]
    swim_mag: Annotated[
        float, Field(ge=170.0, le=230.0, json_schema_extra={"editable": True})
    ]
    swim_freq: Annotated[
        float, Field(ge=1.4, le=2.6, json_schema_extra={"editable": True})
    ]


def dinosaur():
    open_mouth = U() > 0
    # body_size = {
    #     'scale_x': 20 + N(0, 2),
    #     'scale_y': 1,
    #     'scale_z': 1,
    # }
    body_fac = parts.body_tube.ReptileBody(type="dinosaur_body")
    body = genome.part(body_fac)
    shoulder_bounds = np.array([[-20, -20, -20], [20, 20, 20]])

    # fleg_fac = parts.reptile_detail.LizardFrontLeg()
    # toe_fac = parts.reptile_detail.LizardToe()
    # for side in [-1, 1]:
    #     leg = genome.part(fleg_fac)
    #     leg = genome.attach(genome.part(toe_fac), leg, coord=(0.97, 0.5, 0.6), joint=Joint(rest=(0,0,40)))
    #     leg = genome.attach(genome.part(toe_fac), leg, coord=(0.98, 0.5, 0.3), joint=Joint(rest=(0,0,13)))
    #     leg = genome.attach(genome.part(toe_fac), leg, coord=(0.98, 0.5, -0.3), joint=Joint(rest=(0,0,-13)))
    #     leg = genome.attach(genome.part(toe_fac), leg, coord=(0.97, 0.5, -0.6), joint=Joint(rest=(0,0,-40)))
    #     genome.attach(leg, body, coord=(U(0.75, 0.77), 0.5, 0.7), joint=Joint(rest=(0, 0, 110), bounds=shoulder_bounds), side=side)

    # bleg_fac = parts.reptile_detail.LizardBackLeg()
    # for side in [-1, 1]:
    #     leg = genome.part(bleg_fac)
    #     leg = genome.attach(genome.part(toe_fac), leg, coord=(0.97, 0.5, 0.6), joint=Joint(rest=(0,0,40)))
    #     leg = genome.attach(genome.part(toe_fac), leg, coord=(0.98, 0.5, 0.3), joint=Joint(rest=(0,0,13)))
    #     leg = genome.attach(genome.part(toe_fac), leg, coord=(0.98, 0.5, -0.3), joint=Joint(rest=(0,0,-13)))
    #     leg = genome.attach(genome.part(toe_fac), leg, coord=(0.97, 0.5, -0.6), joint=Joint(rest=(0,0,-40)))
    #     genome.attach(leg, body, coord=(U(0.81, 0.83), 0.5, 0.6), joint=Joint(rest=(0, 0, 80), bounds=shoulder_bounds), side=side)

    # neck_fac = parts.reptile_neck.ReptileNeck()
    # neck = genome.part(neck_fac)
    # genome.attach(neck, body, coord=(0.1, 0, 0.2), joint=Joint(rest=(180, 180, 0)), rotation_basis='global', bridge_rad=0.2, smooth_rad=0.1)

    # head_size = {
    #     'scale_x': 0.8 + N(0, 0.02),
    #     'scale_y': 0.3 + N(0, 0.02),
    # }
    # head_fac = parts.reptile_detail.ReptileUpperHead(head_size)
    # head = genome.part(head_fac)
    # genome.attach(head, neck, coord=(0.88, 0, 0.2), joint=Joint(rest=(180, 180, 180)), rotation_basis='global', bridge_rad=0.2, smooth_rad=0.1)

    # eye_fac = parts.eye.MammalEye({'Radius': N(0.03, 0.005)})
    # t, splay = U(0.7, 0.7), 100/180
    # r = 1
    # rot = np.array([0, 0, 90]) * N(1, 0.1, 3)
    # for side in [-1, 1]:
    #     eye = genome.part(eye_fac)
    #     genome.attach(eye, head, coord=(t, splay, r), joint=Joint(rest=(0,0,0)), rotation_basis='normal', side=side)

    # # teeth
    # horn_fac = parts.horn.Horn({'depth_of_ridge': 0, 'length': U(0.2, 0.3), 'rad1': U(0.4, 0.4), 'rad2': U(0.3, 0.3), 'thickness': U(0.04, 0.08), 'height': 0})
    # t, splay = U(0.67, 0.7), 60/180
    # for side in [-1, 1]:
    #     horn = genome.part(horn_fac)
    #     genome.attach(horn, head, coord=(t, splay, 0.8), joint=Joint(rest=(30, 130, -20)), rotation_basis='global', side=side)

    # jaw_fac = parts.reptile_detail.ReptileLowerHead(head_size)
    # jaw = genome.part(jaw_fac)
    # genome.attach(jaw, neck, coord=(0.88, 0, 0.1), joint=Joint(rest=(180, 170, 180)), rotation_basis='global', bridge_rad=0.1, smooth_rad=0.1)

    return genome.CreatureGenome(
        parts=body,
        postprocess_params=dict(
            animation=dict(),
        ),
    )


def lizard_genome(p: LizardParameters | None = None):
    open_mouth = U() > 0

    head_scale_x = p.head_scale_x if p is not None else 0.8 + N(0, 0.02)
    head_scale_y = p.head_scale_y if p is not None else 0.3 + N(0, 0.02)
    head_size = {
        "scale_x": head_scale_x,
        "scale_y": head_scale_y,
    }
    head_fac = parts.reptile_detail.ReptileUpperHead(head_size)
    head = genome.part(head_fac)
    # genome.attach(head, body, coord=(0.01, 0, 0.2), joint=Joint(rest=(180, 180, 0)), rotation_basis='global', bridge_rad=0.2, smooth_rad=0.1)

    # eye_fac = parts.eye.MammalEye({'Radius': N(0.03, 0.005)})
    # t, splay = U(0.7, 0.7), 100/180
    # r = 1
    # rot = np.array([0, 0, 90]) * N(1, 0.1, 3)
    # for side in [-1, 1]:
    #     eye = genome.part(eye_fac)
    #     genome.attach(eye, head, coord=(t, splay, r), joint=Joint(rest=(0,0,0)), rotation_basis='normal', side=side)

    # # teeth
    # horn_fac = parts.horn.Horn({'depth_of_ridge': 0, 'length': U(0.2, 0.3), 'rad1': U(0.4, 0.4), 'rad2': U(0.3, 0.3), 'thickness': U(0.04, 0.08), 'height': 0})
    # t, splay = U(0.67, 0.7), 60/180
    # for side in [-1, 1]:
    #     horn = genome.part(horn_fac)
    #     genome.attach(horn, head, coord=(t, splay, 0.8), joint=Joint(rest=(30, 130, -20)), rotation_basis='global', side=side)

    # jaw_fac = parts.reptile_detail.ReptileLowerHead(head_size)
    # jaw = genome.part(jaw_fac)
    # genome.attach(jaw, body, coord=(0.01, 0, 0.1), joint=Joint(rest=(180, 150, 0)), rotation_basis='global', bridge_rad=0.1, smooth_rad=0.1)

    return genome.CreatureGenome(
        parts=head,
        postprocess_params=dict(
            anim=lizard_run_params(p),
        ),
    )


def snake_genome(p: SnakeParameters | None = None):
    open_mouth = U() > 0

    w_mod = p.width_mod if p is not None else float(N(1, 0.05))
    h_mod = p.height_mod if p is not None else float(N(1, 0.05))

    body_fac = parts.reptile_detail.ReptileBody(
        type="snake",
        n_bones=15,
        shoulder_ik_ts=[0.0, 0.3, 0.6, 1.0],
        mod=(1, w_mod, h_mod),
    )

    body = genome.part(body_fac)

    head_scale_x = p.head_scale_x if p is not None else 0.8 + N(0, 0.02)
    head_size = {
        "scale_x": head_scale_x,
        "scale_y": 0.3 + N(0, 0.02),
    }

    head_fac = parts.reptile_detail.ReptileUpperHead(head_size, mod=(1, w_mod, h_mod))
    head = genome.part(head_fac)
    genome.attach(
        head,
        body,
        coord=(0.01, 0, 0.2),
        joint=Joint(rest=(180, 180, 0)),
        rotation_basis="global",
        bridge_rad=0.2,
        smooth_rad=0.1,
    )

    eye_radius = p.eye_radius if p is not None else float(N(0.03, 0.005))
    eye_fac = parts.eye.MammalEye({"Radius": eye_radius})
    t, splay = U(0.7, 0.7), 100 / 180
    r = 1
    rot = np.array([0, 0, 90]) * N(1, 0.1, 3)
    for side in [-1, 1]:
        eye = genome.part(eye_fac)
        genome.attach(
            eye,
            head,
            coord=(t, splay, r),
            joint=Joint(rest=(0, 0, 0)),
            rotation_basis="normal",
            side=side,
        )

    # teeth
    horn_fac = parts.horn.Horn(
        {
            "depth_of_ridge": 0,
            "length": U(0.2, 0.3),
            "rad1": U(0.4, 0.4),
            "rad2": U(0.3, 0.3),
            "thickness": U(0.04, 0.08),
            "height": 0,
        }
    )
    t, splay = U(0.67, 0.7), 60 / 180
    for side in [-1, 1]:
        horn = genome.part(horn_fac)
        genome.attach(
            horn,
            head,
            coord=(t, splay, 0.8),
            joint=Joint(rest=(30, 130, -20)),
            rotation_basis="global",
            side=side,
        )

    jaw_fac = parts.reptile_detail.ReptileLowerHead(head_size, mod=(1, w_mod, h_mod))
    jaw = genome.part(jaw_fac)
    mouth_open_deg = 0
    genome.attach(
        jaw,
        body,
        coord=(0.01, 0, 0.15),
        joint=Joint(rest=(180, 180 - mouth_open_deg, 0)),
        rotation_basis="global",
        bridge_rad=0.1,
        smooth_rad=0.1,
    )

    return genome.CreatureGenome(
        parts=body,
        postprocess_params=dict(
            anim=snake_swim_params(p),
        ),
    )


def chameleon_genome(p: ChameleonParameters | None = None):
    open_mouth = U() > 0

    body_fac = parts.chameleon.Chameleon()
    body = genome.part(body_fac)

    return genome.CreatureGenome(
        parts=body,
        postprocess_params=dict(
            anim=snake_swim_params(p),
        ),
    )


def frog_genome(p: FrogParameters | None = None):
    # body_fac = parts.reptile_detail.ReptileHeadBody(params={'open_mouth': False}, type='frog')
    # body = genome.part(body_fac)
    # shoulder_bounds = np.array([[-20, -20, -20], [20, 20, 20]])
    # open_mouth = U() > 0
    # body_fac = parts.body_tube.ReptileBody(type='frog_body')
    # body = genome.part(body_fac)
    # shoulder_bounds = np.array([[-20, -20, -20], [20, 20, 20]])

    # fleg_fac = parts.reptile_detail.LizardFrontLeg(type='frog')
    # toe_fac = parts.reptile_detail.LizardToe()
    # for side in [-1, 1]:
    #     leg = genome.part(fleg_fac)
    #     leg = genome.attach(genome.part(toe_fac), leg, coord=(0.97, 0.5, 0.6), joint=Joint(rest=(0,0,40)))
    #     leg = genome.attach(genome.part(toe_fac), leg, coord=(0.98, 0.5, 0.3), joint=Joint(rest=(0,0,13)))
    #     leg = genome.attach(genome.part(toe_fac), leg, coord=(0.98, 0.5, -0.3), joint=Joint(rest=(0,0,-13)))
    #     leg = genome.attach(genome.part(toe_fac), leg, coord=(0.97, 0.5, -0.6), joint=Joint(rest=(0,0,-40)))
    #     genome.attach(leg, body, coord=(U(0.5, 0.55), 0.45, 0.9), joint=Joint(rest=(0, 0, 110), bounds=shoulder_bounds), side=side)

    # bleg_fac = parts.reptile_detail.LizardBackLeg(type='frog')
    # for side in [-1, 1]:
    #     leg = genome.part(bleg_fac)
    #     leg = genome.attach(genome.part(toe_fac), leg, coord=(0.97, 0.5, 0.6), joint=Joint(rest=(0,0,40)))
    #     leg = genome.attach(genome.part(toe_fac), leg, coord=(0.98, 0.5, 0.3), joint=Joint(rest=(0,0,13)))
    #     leg = genome.attach(genome.part(toe_fac), leg, coord=(0.98, 0.5, -0.3), joint=Joint(rest=(0,0,-13)))
    #     leg = genome.attach(genome.part(toe_fac), leg, coord=(0.97, 0.5, -0.6), joint=Joint(rest=(0,0,-40)))
    #     genome.attach(leg, body, coord=(U(0.70, 0.75), 0.45, 0.8), joint=Joint(rest=(0, 0, 50), bounds=shoulder_bounds), side=side)

    head_scale_y = p.head_scale_y if p is not None else 0.4 + N(0, 0.02)
    head_scale_x = p.head_scale_x if p is not None else 0.8 + N(0, 0.02)
    head_size = {
        "scale_x": head_scale_x,
        "scale_y": head_scale_y,
    }
    head_fac = parts.reptile_detail.ReptileUpperHead(head_size)
    head = genome.part(head_fac)
    # genome.attach(head, body, coord=(0.1, 0.5, 0.5), joint=Joint(rest=(180, 180, 0)), rotation_basis='global')

    # eye_fac = parts.eye.MammalEye({'Radius': N(0.03, 0.005)})
    # t, splay = U(0.7, 0.7), 100/180
    # r = 1
    # rot = np.array([0, 0, 90]) * N(1, 0.1, 3)
    # for side in [-1, 1]:
    #     eye = genome.part(eye_fac)
    #     genome.attach(eye, head, coord=(t, splay, r), joint=Joint(rest=(0,0,0)), rotation_basis='normal', side=side)

    # jaw_fac = parts.reptile_detail.ReptileLowerHead(head_size)
    # jaw = genome.part(jaw_fac)
    # genome.attach(jaw, body, coord=(0.1, 0.5, 0.5), joint=Joint(rest=(180, 170, 0)), rotation_basis='global')

    speed_m_s = p.speed_m_s if p is not None else 0.5
    return genome.CreatureGenome(
        parts=head,
        postprocess_func=reptile_postprocessing,
        postprocess_params=dict(
            animation=dict(mode="swim", speed_m_s=speed_m_s),
        ),
    )


def snake_swim_params(p: SnakeParameters | ChameleonParameters | None = None):
    swim_freq = (
        p.swim_freq
        if isinstance(p, ChameleonParameters)
        else 2 * clip_gaussian(1, 0.3, 0.1, 2)
    )
    swim_mag = p.swim_mag if p is not None else float(N(200, 3))
    return dict(
        swim_mag=swim_mag,
        swim_freq=swim_freq,
        flipper_freq=2 * clip_gaussian(1, 0.5, 0.1, 3) * swim_freq,
        flipper_mag=0.25 * N(1, 0.1) * swim_mag,
        flipper_var=U(0, 0.2),
    )


def chameleon_eye_params():
    swim_freq = 0.2 * clip_gaussian(1, 0.3, 0.1, 2)
    swim_mag = N(20, 3)
    return dict(
        swim_mag=swim_mag,
        swim_freq=swim_freq,
        flipper_freq=2 * clip_gaussian(1, 0.5, 0.1, 3) * swim_freq,
        flipper_mag=0.25 * N(1, 0.1) * swim_mag,
        flipper_var=U(0, 0.2),
    )


def animate_snake_swim(root, arma, params, ik_targets):
    spine = [b for b in arma.pose.bones if "Body" in b.name]
    creature_animation.animate_wiggle_bones(
        arma=arma,
        bones=spine,
        fixed_head=False,
        off=1 / 2,
        mag_deg=params["swim_mag"],
        freq=params["swim_freq"],
        wavelength=U(0.2, 0.4),
    )


def animate_chameleon_eye(root, arma, params, ik_targets):
    spine = [b for b in arma.pose.bones if "Eye" in b.name]
    creature_animation.animate_wiggle_bones(
        arma=arma,
        bones=spine,
        fixed_head=False,
        off=1 / 2,
        mag_deg=params["swim_mag"],
        freq=params["swim_freq"],
        wavelength=U(0.2, 0.4),
    )


def lizard_run_params(p: LizardParameters | None = None):
    swim_freq = (
        p.swim_freq if p is not None else float(clip_gaussian(1, 0.3, 0.1, 2))
    )
    swim_mag = p.swim_mag if p is not None else float(N(50, 3))
    return dict(
        swim_mag=swim_mag,
        swim_freq=swim_freq,
        flipper_freq=2 * clip_gaussian(1, 0.5, 0.1, 3) * swim_freq,
        flipper_mag=0.25 * N(1, 0.1) * swim_mag,
        flipper_var=U(0, 0.2),
    )


def animate_lizard_run(root, arma, params, ik_targets):
    spine = [b for b in arma.pose.bones if "Body" in b.name]
    creature_animation.animate_wiggle_bones(
        arma=arma,
        bones=spine,
        fixed_head=False,
        off=1 / 2,
        mag_deg=params["swim_mag"],
        freq=params["swim_freq"],
        wavelength=U(1, 1.2),
    )

    spine = [b for b in arma.pose.bones if "FrontLeg" in b.name]
    print(spine)
    creature_animation.animate_running_front_leg(
        arma=arma,
        bones=spine,
        fixed_head=False,
        off=1 / 2,
        mag_deg=params["swim_mag"],
        freq=params["swim_freq"],
        wavelength=U(1, 1.2),
    )

    spine = [b for b in arma.pose.bones if "BackLeg" in b.name]
    print(spine)
    creature_animation.animate_running_back_leg(
        arma=arma,
        bones=spine,
        fixed_head=False,
        off=0,
        mag_deg=params["swim_mag"],
        freq=params["swim_freq"],
        wavelength=U(1, 1.2),
    )
    # creature_animation.animate_run(root, arma, ik_targets)


def reptile_postprocessing(body_parts, extras, params):
    def get_extras(k):
        return [o for o in extras if k in o.name]

    main_template = weighted_sample(material_assignments.reptile)
    body = body_parts + get_extras("BodyExtra")
    main_template.apply(body)

    tongue.apply(get_extras("Tongue"))
    bone.apply(get_extras("Horn"))
    eyeball.apply(get_extras("Eyeball"), shader_kwargs={"coord": "X"})
    nose.apply(get_extras("Nose"))


def chameleon_postprocessing(body_parts, extras, params):
    def get_extras(k):
        return [o for o in extras if k in o.name]

    main_template = weighted_sample(material_assignments.reptile)
    body = body_parts + get_extras("BodyExtra")
    main_template.apply(body)

    # chameleon_eye.apply(get_extras('Eye'))


@gin.configurable
class LizardFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = LizardParameters
    max_distance = 40

    def __init__(self, factory_seed, bvh=None, coarse=False):
        self.bvh = bvh
        super().__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> LizardParameters:
        return LizardParameters(
            seed=seed,
            head_scale_x=float(np.clip(0.8 + N(0, 0.02), 0.76, 0.84)),
            head_scale_y=float(np.clip(0.3 + N(0, 0.02), 0.26, 0.34)),
            swim_mag=float(np.clip(N(50, 3), 41.0, 59.0)),
            swim_freq=float(np.clip(clip_gaussian(1, 0.3, 0.1, 2), 0.7, 1.3)),
        )

    def apply_parameters(
        self, params: LizardParameters, *, spawn_scope: bool = True
    ) -> None:
        self._lizard_params = params
        self._use_fixed_spawn_draws = spawn_scope

    def create_asset(self, i, animate=False, rigging=False, cloth=False, **kwargs):
        lizard_params = self._lizard_params if self._use_fixed_spawn_draws else None
        genome = lizard_genome(lizard_params)
        root, parts = creature.genome_to_creature(
            genome, name=f"lizard({self.factory_seed}, {i})"
        )

        joined, extras, arma, ik_targets = joining.join_and_rig_parts(
            root,
            parts,
            genome,
            postprocess_func=reptile_postprocessing,
            adapt_mode="remesh",
            rigging=rigging,
            **kwargs,
        )
        if animate and arma is not None:
            pass
        else:
            joined = butil.join_objects([joined] + extras)

        butil.purge_empty_materials(joined)

        return root


@gin.configurable
class FrogFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = FrogParameters
    max_distance = 40

    def __init__(self, factory_seed, bvh=None, coarse=False):
        self.bvh = bvh
        super().__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> FrogParameters:
        return FrogParameters(
            seed=seed,
            head_scale_x=float(0.8 + N(0, 0.02)),
            head_scale_y=float(0.4 + N(0, 0.02)),
            speed_m_s=0.5,
        )

    def apply_parameters(
        self, params: FrogParameters, *, spawn_scope: bool = True
    ) -> None:
        self._frog_params = params
        self._use_fixed_spawn_draws = spawn_scope

    def create_asset(self, i, animate=False, rigging=False, simulate=False, **kwargs):
        frog_params = self._frog_params if self._use_fixed_spawn_draws else None
        genome = frog_genome(frog_params)
        root, parts = creature.genome_to_creature(
            genome, name=f"frog({self.factory_seed}, {i})"
        )

        joined, extras, arma, ik_targets = joining.join_and_rig_parts(
            root,
            parts,
            genome,
            postprocess_func=reptile_postprocessing,
            adapt_mode="remesh",
            rigging=rigging,
            **kwargs,
        )
        if animate and arma is not None:
            pass
        if simulate:
            pass
        else:
            joined = butil.join_objects([joined] + extras)

        return root


@gin.configurable
class SnakeFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = SnakeParameters
    max_distance = 40

    def __init__(
        self,
        factory_seed,
        bvh=None,
        coarse=False,
        snake_length=("uniform", 0.5, 3),
        **kwargs,
    ):
        self.bvh = bvh
        self._snake_length_dist = snake_length
        super().__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> SnakeParameters:
        with FixedSeed(seed):
            snake_length = float(random_general(self._snake_length_dist))
        return SnakeParameters(
            seed=seed,
            snake_length=snake_length,
            width_mod=float(N(1, 0.05)),
            height_mod=float(N(1, 0.05)),
            head_scale_x=float(0.8 + N(0, 0.02)),
            eye_radius=float(N(0.03, 0.005)),
            swim_mag=float(N(200, 3)),
        )

    def apply_parameters(
        self, params: SnakeParameters, *, spawn_scope: bool = True
    ) -> None:
        self._snake_params = params
        self.snake_length = params.snake_length
        self.policy = animation_policy.AnimPolicyRandomForwardWalk(
            forward_vec=(1, 0, 0),
            speed=min(self.snake_length, 2) * U(0.5, 1),
            step_range=(0.2, 0.2),
            yaw_dist=("uniform", -7, 7),
        )
        self._use_fixed_spawn_draws = spawn_scope

    def create_placeholder(self, i, loc, rot, **kwargs):
        p = butil.spawn_cube(size=self.snake_length)
        p.location = loc
        p.rotation_euler = rot

        if self.bvh is None:
            return p

        curve = animation_policy.policy_create_bezier_path(
            p, self.bvh, self.policy, eval_offset=(0, 0, 0.5), retry_rotation=True
        )
        curve.name = f"animhelper:{self}.create_placeholder({i}).path"

        slither_curve = butil.deep_clone_obj(curve)
        curve_slither.add_curve_slithers(slither_curve, snake_length=self.snake_length)

        if slither_curve.type != "CURVE":
            logging.warning(
                f"{self.__class__.__name__} created invalid path {curve.name} with {curve.type=}"
            )
            return p

        curve_slither.snap_curve_to_floor(slither_curve, self.bvh)
        butil.parent_to(curve, slither_curve, keep_transform=True)

        # animate the placeholder to the APPROX location of the snake, so the camera can follow it
        follow_path(
            p,
            curve,
            use_curve_follow=True,
            offset=0,
            duration=bpy.context.scene.frame_end - bpy.context.scene.frame_start,
        )
        curve.data.driver_add("eval_time").driver.expression = "frame"

        return p

    def create_asset(self, i, placeholder, **kwargs):
        snake_params = self._snake_params if self._use_fixed_spawn_draws else None
        genome = snake_genome(snake_params)
        root, parts = creature.genome_to_creature(
            genome, name=f"snake({self.factory_seed}, {i})"
        )

        joined, extras, arma, ik_targets = joining.join_and_rig_parts(
            root,
            parts,
            genome,
            postprocess_func=reptile_postprocessing,
            adaptive_resolution=False,
            rigging=False,
            **kwargs,
        )

        joined = butil.join_objects([joined] + extras)

        s = (
            self.snake_length / 20
        )  # convert to real units. existing code averages 20m length
        joined.scale = (s, s, s)
        butil.apply_transform(joined, scale=True)

        if (
            len(placeholder.constraints)
            and placeholder.constraints[0].type == "FOLLOW_PATH"
        ):
            curve = placeholder.constraints[0].target.parent
            assert curve.type == "CURVE", curve.type
            if len(curve.data.splines[0].points) > 3:
                orig_len = curve.data.splines[0].calc_length()

                joined.parent = None
                curve_slither.slither_along_path(
                    joined, curve, speed=self.policy.speed, orig_len=orig_len
                )

                root.parent = butil.spawn_empty(
                    "snake_parent_temp"
                )  # so AssetFactory.spawn_asset doesnt attempt to parent
                butil.parent_to(joined, root, keep_transform=True)

        butil.purge_empty_materials(joined)

        return joined


@gin.configurable
class ChameleonFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = ChameleonParameters
    max_distance = 40

    def __init__(self, factory_seed, bvh=None, coarse=False, **kwargs):
        self.bvh = bvh
        super().__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> ChameleonParameters:
        return ChameleonParameters(
            seed=seed,
            body_scale=float(N(1, 0.1)),
            swim_mag=float(N(200, 3)),
            swim_freq=float(2 * clip_gaussian(1, 0.3, 0.1, 2)),
        )

    def apply_parameters(
        self, params: ChameleonParameters, *, spawn_scope: bool = True
    ) -> None:
        self._chameleon_params = params
        self._use_fixed_spawn_draws = spawn_scope

    def create_placeholder(self, i, loc, rot, **kwargs):
        p = butil.spawn_cube(size=1)
        p.location = loc
        p.rotation_euler = rot

        return p

    def create_asset(self, i, placeholder, **kwargs):
        chameleon_params = (
            self._chameleon_params if self._use_fixed_spawn_draws else None
        )
        genome = chameleon_genome(chameleon_params)
        root, parts = creature.genome_to_creature(
            genome, name=f"snake({self.factory_seed}, {i})"
        )

        joined, extras, arma, ik_targets = joining.join_and_rig_parts(
            root,
            parts,
            genome,
            postprocess_func=reptile_postprocessing,
            adaptive_resolution=False,
            rigging=False,
            **kwargs,
        )

        joined = butil.join_objects([joined] + extras)
        if chameleon_params is not None:
            joined.scale = (
                chameleon_params.body_scale,
                chameleon_params.body_scale,
                chameleon_params.body_scale,
            )
            butil.apply_transform(joined, scale=True)

        return root
