"""Task configuration (what aerial_gym's BaseTask and NavigationTask read)."""

from wmnav.env.configs.robot import LMF2Cfg


def make_task_cfg(mode: str, num_envs: int, seed: int, device: str = "cuda:0", headless: bool = True,
                  episode_len_steps: int = 300, sim_name: str = "wmnav_sim", env_name: str = "wmnav_env",
                  robot_name: str = "wmnav_lmf2"):
    """The aerial_gym task config of one run: reset ``mode`` (``world_model`` or
    ``ppo``), env count, seed, device, episode length and the registered names.
    """
    cam = LMF2Cfg.sensor_config.camera_config

    class NavTaskCfg:  # attribute namespace, filled in below
        pass

    c = NavTaskCfg
    c.mode = mode                         # "world_model" | "ppo"
    c.seed = seed
    c.sim_name, c.env_name, c.robot_name = sim_name, env_name, robot_name
    c.controller_name = "lmf2_velocity_control"
    c.args = {}
    c.num_envs = num_envs
    c.use_warp = True
    c.headless = headless
    c.device = device
    c.privileged_observation_space_dim = 0
    c.action_space_dim = 4
    c.observation_space_dim = 22          # "pos rot unit vector" robot state
    c.camera_shape = (cam.height, cam.width, 1)
    c.episode_len_steps = episode_len_steps   # env steps of 0.1 s
    c.return_state_before_reset = True
    c.preprocess_actions = True           # agent action -> velocity command (task.transform_action)
    c.target_x, c.target_y, c.target_z = (13.0, 14.0), (-2.0, 2.0), (1.0, 5.0)
    c.reward_parameters = {
        "a1": 2.0, "b1": 1.0,      # close to the target
        "a2": 2.0, "b2": 0.008,    # far from the target
        "a3": 0.0,                 # crash
        "a4": 0.0,                 # win
        "a5": 1.0,                 # win * remaining steps
        "dist_thr": 0.5,           # [m] win radius
    }
    return c
