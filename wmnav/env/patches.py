"""Performance patches for aerial_gym (luca_dev @ dcd757dd), applied by wmnav only.

aerial_gym logs tensors with ``logger.debug(f"...{tensor}...")``. Python builds the
f-string - formatting the CUDA tensors, i.e. copying them to the host - before
the logger discards the message, so every call stalls the GPU even with debug
logging off. Two such calls sat in the per-physics-step path and caused ~1,570
host synchronisations per environment step; three more format whole tensors on
every reset (the robot's random initial state, the env bounds, the reset ids).

The methods below are verbatim copies of aerial_gym's with only that logging
removed, so the simulation is unchanged: tests/test_env_rollout.py compares
rollouts against the unpatched legacy code (bit-identical). The aerial_gym
sources themselves are not modified.

Derived from Aerial Gym Simulator (https://github.com/ntnu-arl/aerial_gym_simulator,
BSD 3-Clause License, Copyright (c) 2023 Autonomous Robots Lab, NTNU); see
THIRD_PARTY_NOTICES.md.
"""

import aerial_gym.control.controllers.base_lee_controller as _lee
import aerial_gym.env_manager.env_manager as _env_manager
import aerial_gym.robots.base_multirotor as _multirotor
import aerial_gym.sensors.warp.warp_sensor as _warp

# the helpers exactly as the original methods resolve them (aerial_gym.utils.math
# defines some names twice; these are the bindings the modules actually see)
quat_rotate = _lee.quat_rotate
tf_apply, quat_mul = _warp.tf_apply, _warp.quat_mul
torch_rand_float_tensor = _multirotor.torch_rand_float_tensor
torch_interpolate_ratio = _multirotor.torch_interpolate_ratio
quat_from_euler_xyz_tensor = _multirotor.quat_from_euler_xyz_tensor
BaseLeeController, WarpSensor = _lee.BaseLeeController, _warp.WarpSensor
BaseMultirotor, EnvManager = _multirotor.BaseMultirotor, _env_manager.EnvManager

_APPLIED = False


def _compute_acceleration(self, setpoint_position, setpoint_velocity):
    position_error_world_frame = setpoint_position - self.robot_position
    setpoint_velocity_world_frame = quat_rotate(self.robot_vehicle_orientation, setpoint_velocity)
    velocity_error = setpoint_velocity_world_frame - self.robot_linvel
    accel_command = (
        self.K_pos_tensor_current * position_error_world_frame
        + self.K_linvel_tensor_current * velocity_error
    )
    return accel_command


def _warp_update(self):
    self.sensor_position[:] = tf_apply(self.robot_orientation, self.robot_position, self.sensor_local_position)
    self.sensor_orientation[:] = quat_mul(
        self.robot_orientation,
        quat_mul(self.sensor_local_orientation, self.sensor_data_frame_quat),
    )
    self.sensor.capture()
    self.apply_noise()
    if self.cfg.sensor_type in ["camera", "lidar"]:
        self.apply_range_limits()
        self.normalize_observation()


def _multirotor_reset_idx(self, env_ids):
    if len(env_ids) == 0:
        return
    random_state = torch_rand_float_tensor(self.min_init_state, self.max_init_state)
    self.robot_state[env_ids, 0:3] = torch_interpolate_ratio(
        self.env_bounds_min, self.env_bounds_max, random_state[:, 0:3]
    )[env_ids]
    self.robot_state[env_ids, 3:7] = quat_from_euler_xyz_tensor(random_state[env_ids, 3:6])
    self.robot_state[env_ids, 7:10] = random_state[env_ids, 7:10]
    self.robot_state[env_ids, 10:13] = random_state[env_ids, 10:13]
    self.controller.randomize_params(env_ids=env_ids)
    self.update_states()


def _env_manager_reset_idx(self, env_ids=None):
    self.IGE_env.reset_idx(env_ids)
    self.asset_manager.reset_idx(env_ids, self.global_tensor_dict["num_obstacles_in_env"])
    if self.cfg.env.use_warp:
        self.warp_env.reset_idx(env_ids)
    self.robot_manager.reset_idx(env_ids)
    self.IGE_env.write_to_sim()
    self.sim_steps[env_ids] = 0


_PATCHES = (
    (BaseLeeController, "compute_acceleration", _compute_acceleration),
    (WarpSensor, "update", _warp_update),
    (BaseMultirotor, "reset_idx", _multirotor_reset_idx),
    (EnvManager, "reset_idx", _env_manager_reset_idx),
)


def apply() -> None:
    """Install the patched methods on aerial_gym's classes (idempotent)."""
    global _APPLIED
    if _APPLIED:
        return
    for cls, name, method in _PATCHES:
        setattr(cls, name, method)
    _APPLIED = True
