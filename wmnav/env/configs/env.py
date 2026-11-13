"""Environment (scene) configuration and the per-run builder.

``build_env_cfg(level, num_panels)`` returns fresh subclasses for one run, so a
level never mutates shared configuration (the legacy code edited the config
files, or patched the classes in place, between runs).
"""

from wmnav.env.configs import assets
from wmnav.env.levels import level_spec


class NavEnvCfg:
    """aerial_gym environment config of the corridor (aerial_gym reads the nested
    classes by these lowercase names).
    """

    class env:
        """Simulation-wide environment settings and the corridor bounds."""

        num_envs = 16                 # overridden by the task's num_envs
        num_env_actions = 6
        env_spacing = 5.0
        num_physics_steps_per_env_step_mean = 10   # 10 x 0.01 s = 0.1 s per env step (10 Hz)
        num_physics_steps_per_env_step_std = 0
        render_viewer_every_n_steps = 1
        reset_on_collision = True
        collision_force_threshold = 0.05           # [N] contact force counted as a crash
        create_ground_plane = False
        sample_timestep_for_latency = False
        perturb_observations = False
        keep_same_env_for_num_episodes = 1
        write_to_sim_at_every_timestep = True
        use_warp = False
        # bounds and obstacle distribution: set per level by build_env_cfg
        lower_bound_min = None
        lower_bound_max = None
        upper_bound_min = None
        upper_bound_max = None
        distribution = None

    class env_config:
        """Which assets are loaded, in load order."""

        include_asset_type = {
            "left_wall": True, "right_wall": True, "back_wall": True, "front_wall": True,
            "bottom_wall": True, "top_wall": True, "pillars": True,
        }
        # order matters: walls are loaded first, and analyses index them as assets 0-5
        asset_type_to_dict_map = {
            "left_wall": assets.left_wall,
            "right_wall": assets.right_wall,
            "back_wall": assets.back_wall,
            "front_wall": assets.front_wall,
            "bottom_wall": assets.bottom_wall,
            "top_wall": assets.top_wall,
            "pillars": assets.pillar_params,
        }


def build_env_cfg(level: str, num_panels: int):
    """A fresh environment config for ``level`` with ``num_panels`` obstacle panels."""
    spec = level_spec(level)
    pillars = type("pillar_params", (assets.pillar_params,), {"num_assets": int(num_panels)})
    env = type("env", (NavEnvCfg.env,), dict(spec["bounds"], distribution=spec["distribution"]))
    mapping = dict(NavEnvCfg.env_config.asset_type_to_dict_map, pillars=pillars)
    env_config = type("env_config", (NavEnvCfg.env_config,), {"asset_type_to_dict_map": mapping})
    return type(f"NavEnvCfg_{level}_{num_panels}", (), {"env": env, "env_config": env_config})
