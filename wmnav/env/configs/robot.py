"""LMF2 quadrotor with the forward depth camera, as flown in the paper's experiments.

Controller gains (``lmf2_velocity_control``) are aerial_gym's own
``lmf2_controller_config``: randomised uniformly per episode within its ranges.
Motor thrust constant and spin-up time constant are randomised by the motor model
below. Disturbances and aerodynamic damping are off.

Derived from Aerial Gym Simulator (https://github.com/ntnu-arl/aerial_gym_simulator,
BSD 3-Clause License, Copyright (c) 2023 Autonomous Robots Lab, NTNU); see
THIRD_PARTY_NOTICES.md.
"""

import numpy as np
from aerial_gym import AERIAL_GYM_DIRECTORY
from aerial_gym.config.sensor_config.camera_config.base_depth_camera_config import BaseDepthCameraConfig
from aerial_gym.config.sensor_config.imu_config.base_imu_config import BaseImuConfig
from aerial_gym.config.sensor_config.lidar_config.base_lidar_config import BaseLidarConfig

# [ratio_x, ratio_y, ratio_z, roll, pitch, yaw, 1.0, vx, vy, vz, wx, wy, wz]; positions
# are ratios of the environment bounds
_SPAWN_MIN = [0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 1.0, 0, 0, 0, 0, 0, 0]
_SPAWN_MAX = [0.2, 0.9, 0.9, 0.0, 0.0, 0.0, 1.0, 0, 0, 0, 0, 0, 0]


class LMF2Cfg:
    """aerial_gym robot config of the LMF2 quadrotor (nested classes are read by name)."""

    class init_config:
        min_init_state = list(_SPAWN_MIN)  # spawn in the first 10-20 % of the corridor, at rest
        max_init_state = list(_SPAWN_MAX)

    class sensor_config:  # only the depth camera is on
        enable_camera = True
        camera_config = BaseDepthCameraConfig  # 64x96 depth, 87 deg HFOV, 0.2-10 m, normalised
        enable_lidar = False
        lidar_config = BaseLidarConfig
        enable_imu = False
        imu_config = BaseImuConfig

    class disturbance:  # random external forces (off)
        enable_disturbance = False
        prob_apply_disturbance = 0.05
        max_force_and_torque_disturbance = [4.75, 4.75, 4.75, 0.03, 0.03, 0.03]

    class damping:  # aerodynamic damping (off)
        linvel_linear_damping_coefficient = [0.0, 0.0, 0.0]
        linvel_quadratic_damping_coefficient = [0.0, 0.0, 0.0]
        angular_linear_damping_coefficient = [0.0, 0.0, 0.0]
        angular_quadratic_damping_coefficient = [0.0, 0.0, 0.0]

    class robot_asset:  # URDF and Isaac Gym asset options
        asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/robots/lmf2"
        file = "model.urdf"
        name = "base_quadrotor"
        base_link_name = "base_link"
        disable_gravity = False
        collapse_fixed_joints = True
        fix_base_link = False
        collision_mask = 0
        replace_cylinder_with_capsule = False
        flip_visual_attachments = True
        density = 0.000001
        angular_damping = 0.01
        linear_damping = 0.01
        max_angular_velocity = 100.0
        max_linear_velocity = 100.0
        armature = 0.001
        semantic_id = 0
        per_link_semantic = False
        min_state_ratio = [0.5, 0.1, 0.5, 0, 0, np.pi / 2, 1.0, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2]
        max_state_ratio = [0.5, 0.1, 0.5, 0, 0, np.pi / 2, 1.0, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2]
        max_force_and_torque_disturbance = [0.1, 0.1, 0.1, 0.05, 0.05, 0.05]
        color = None
        semantic_masked_links = {}
        keep_in_env = True
        min_position_ratio = None
        max_position_ratio = None
        min_euler_angles = [-np.pi, -np.pi, -np.pi]
        max_euler_angles = [np.pi, np.pi, np.pi]
        place_force_sensor = True
        force_sensor_parent_link = "base_link"
        force_sensor_transform = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        use_collision_mesh_instead_of_visual = False

    class control_allocator_config:  # motor layout: thrusts -> body force / torque
        num_motors = 4
        force_application_level = "base_link"
        application_mask = [1 + 4 + i for i in range(4)]
        motor_directions = [1, -1, 1, -1]
        allocation_matrix = [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [-0.13, -0.13, 0.13, 0.13],
            [-0.13, 0.13, 0.13, -0.13],
            [-0.07, 0.07, -0.07, 0.07],
        ]

        class motor_model_config:  # first-order motor dynamics
            use_rps = True
            motor_thrust_constant_min = 0.00000926312
            motor_thrust_constant_max = 0.00001826312
            motor_time_constant_increasing_min = 0.05
            motor_time_constant_increasing_max = 0.08
            motor_time_constant_decreasing_min = 0.005
            motor_time_constant_decreasing_max = 0.005
            max_thrust = 10.0
            min_thrust = 0.1
            max_thrust_rate = 100000.0
            thrust_to_torque_ratio = 0.07
            use_discrete_approximation = True
