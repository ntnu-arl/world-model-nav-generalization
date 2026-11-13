"""Physics settings. One environment step = 10 physics steps of 0.01 s (see env.py),
i.e. the agent acts and observes at 10 Hz.

Derived from Aerial Gym Simulator (https://github.com/ntnu-arl/aerial_gym_simulator,
BSD 3-Clause License, Copyright (c) 2023 Autonomous Robots Lab, NTNU); see
THIRD_PARTY_NOTICES.md.
"""


class SimCfg:
    """aerial_gym simulator config (nested classes are read by name)."""

    class viewer:
        headless = False
        ref_env = 0
        camera_position = [-5, -5, 4]
        lookat = [0, 0, 0]
        camera_orientation_euler_deg = [0, 0, 0]
        camera_follow_type = "FOLLOW_TRANSFORM"
        width = 1280
        height = 720
        max_range = 100.0
        min_range = 0.1
        horizontal_fov_deg = 90
        use_collision_geometry = False
        camera_follow_transform_local_offset = [-1.0, 0.0, 0.3]
        camera_follow_position_global_offset = [-1.0, 0.0, 0.3]

    class sim:
        dt = 0.01                    # [s] physics step
        substeps = 1
        gravity = [0.0, 0.0, -9.81]
        up_axis = 1                  # z up
        use_gpu_pipeline = True

        class physx:
            num_threads = 10
            solver_type = 1          # TGS
            num_position_iterations = 4
            num_velocity_iterations = 1
            contact_offset = 0.002
            rest_offset = 0.001
            bounce_threshold_velocity = 0.1
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 24
            default_buffer_size_multiplier = 10
            contact_collection = 1   # last sub-step
