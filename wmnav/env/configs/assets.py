"""Environment assets: six walls closing the corridor and the obstacle panels.

The obstacles ("pillars") are ``big_panel.urdf`` boxes of 0.1 x 1.2 x 50 m,
axis-aligned and taller than the corridor, so the free space is effectively 2-D.
Their x/y placement is set per level (``wmnav.env.levels``); x stays within
20-80 % of the corridor length.

Derived from Aerial Gym Simulator (https://github.com/ntnu-arl/aerial_gym_simulator,
BSD 3-Clause License, Copyright (c) 2023 Autonomous Robots Lab, NTNU); see
THIRD_PARTY_NOTICES.md.
"""

from aerial_gym import AERIAL_GYM_DIRECTORY

FRONT_WALL_SEMANTIC_ID = 9
BACK_WALL_SEMANTIC_ID = 10
LEFT_WALL_SEMANTIC_ID = 11
RIGHT_WALL_SEMANTIC_ID = 12
BOTTOM_WALL_SEMANTIC_ID = 13
TOP_WALL_SEMANTIC_ID = 14

WALLS_FOLDER = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/walls"


class asset_state_params:
    """aerial_gym asset defaults shared by walls and panels (aerial_gym reads this
    class by name).
    """

    num_assets = 10
    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets"
    file = None
    min_position_ratio = [0.5, 0.5, 0.5]
    max_position_ratio = [0.5, 0.5, 0.5]
    collision_mask = 1
    disable_gravity = True
    replace_cylinder_with_capsule = True
    flip_visual_attachments = True
    density = 0.001
    angular_damping = 0.1
    linear_damping = 0.1
    max_angular_velocity = 100.0
    max_linear_velocity = 100.0
    armature = 0.001
    collapse_fixed_joints = True
    fix_base_link = False
    specific_filepath = None
    color = None
    keep_in_env = False
    body_semantic_label = 0
    link_semantic_label = 0
    per_link_semantic = False
    semantic_masked_links = {}
    place_force_sensor = False
    force_sensor_parent_link = "base_link"
    force_sensor_transform = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # position, quat x y z w
    use_collision_mesh_instead_of_visual = False


def _wall(name, file, xyz_ratio, semantic_id, color):
    state = list(xyz_ratio) + [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return type(name, (asset_state_params,), {
        "num_assets": 1,
        "asset_folder": WALLS_FOLDER,
        "file": file,
        "collision_mask": 1,
        "min_state_ratio": list(state),
        "max_state_ratio": list(state),
        "keep_in_env": True,
        "collapse_fixed_joints": True,
        "specific_filepath": "cube.urdf",
        "per_link_semantic": False,
        "semantic_id": semantic_id,
        "color": color,
        "fix_base_link": True,
    })


left_wall = _wall("left_wall", "left_wall.urdf", (0.5, 1.0, 0.5), LEFT_WALL_SEMANTIC_ID, [100, 200, 210])
right_wall = _wall("right_wall", "right_wall.urdf", (0.5, 0.0, 0.5), RIGHT_WALL_SEMANTIC_ID, [100, 200, 210])
back_wall = _wall("back_wall", "back_wall.urdf", (0.0, 0.5, 0.5), BACK_WALL_SEMANTIC_ID, [100, 200, 210])
front_wall = _wall("front_wall", "front_wall.urdf", (1.0, 0.5, 0.5), FRONT_WALL_SEMANTIC_ID, [100, 200, 210])
bottom_wall = _wall("bottom_wall", "bottom_wall.urdf", (0.5, 0.5, 0.0), BOTTOM_WALL_SEMANTIC_ID, [100, 150, 150])
top_wall = _wall("top_wall", "top_wall.urdf", (0.5, 0.5, 1.0), TOP_WALL_SEMANTIC_ID, [100, 200, 210])


class pillar_params(asset_state_params):
    """The obstacle panels; ``num_assets`` and placement are set per level."""

    num_assets = 5  # the training density; levels.apply_level sets it per run
    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/big_panels"
    file = "big_panel.urdf"
    collision_mask = 1
    min_state_ratio = [0.2, 0.0, 0.0, 0, 0, 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    max_state_ratio = [0.8, 1.0, 0.0, 0, 0, 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    keep_in_env = False
    fix_base_link = True
    per_link_semantic = False
    semantic_id = -1  # assigned per instance
