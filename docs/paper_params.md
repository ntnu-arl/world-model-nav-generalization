# Paper parameters

Every number of the paper's setup, read off the legacy configs and the run
artefacts (resolved W&B configs, Hydra records, checkpoints, datasets). `source`
names where a value comes from, so each line can be re-checked; source paths refer
to the legacy repositories `wms_navigation` / `ssl_wms`, with the wmnav location
in brackets where it differs. Lines marked UNVERIFIED / WRONG / DEAD need a
decision before they go in the paper. Parameter names are those of the legacy
configs; [migration](migration.md#dreamer-rewrite-names) maps them to the keys of
`configs/model/dreamer.yaml`.

## Time

timestep: 0.1 s
timestep_source: sim.dt 0.01 (configs/sims/procedural_world.py) x num_physics_steps_per_env_step_mean 10, std 0 (configs/envs/static_env.py)
observation_rate: 10 Hz
observation_rate_note: identical for every stream; the low_freq/high_freq dataset filenames are NOT sampling rates, see dataset section
episode_len_steps: 300
episode_len_seconds: 30.0
episode_len_note: WRONG COMMENT in configs/tasks/static_env.py:19 says episode time = episode_len_steps * sim.dt (= 3 s); it omits the decimation, the real value is 30 s
batch_length: 64
batch_length_seconds: 6.4
batch_length_swept: [16, 32, 64]
batch_length_swept_seconds: [1.6, 3.2, 6.4]
imag_horizon: 15
imag_horizon_seconds: 1.5
discount: 0.997
discount_lambda: 0.95

## Observation

depth_image_resolution: 64 x 96 (H x W), 1 channel
depth_image_source: BaseDepthCameraConfig (aerial_gym/config/sensor_config/camera_config/base_depth_camera_config.py)
depth_image_hfov_deg: 87.0
depth_image_range_m: [0.2, 10.0]
depth_image_normalisation: normalize_range=True -> [0, 1], with -1 for near-out-of-range
depth_image_centering: -0.5 applied inside ConvEncoder.forward, not in preprocess
robot_state_dim: 22
robot_state_layout: "pos rot unit vector" (configs/tasks/static_env.py)
robot_state_preprocess: RunningStandardScaler (world_models/models.py:278)
action_dim: 4
config_wms_size_key: DEAD - `size: [96, 54]` in config_wms.yaml is never read anywhere in the codebase (DreamerV3 boilerplate). Do not quote 54.

## Reconstruction / prediction losses

loss_depth_image: MSE, summed over pixels
loss_depth_image_impl: MSEDist(agg="sum"), decoder image_dist=mse
loss_depth_image_target: depth - 0.5, not depth - an in-place `obs -= 0.5` in the encoder also shifted the batch tensor used as the target (SSL and fine-tuning alike) [wmnav: model.legacy_depth_target_shift]
ssim_note: the paper's SSIM values were computed on depth - 0.5 images (same shift); SSIM is not shift-invariant [wmnav.imagine depth_space=raw gives true-depth SSIM]
loss_robot_state: symlog MSE, summed over dims
loss_robot_state_impl: SymlogDist(dist="mse", agg="sum"), decoder vector_dist=symlog_mse, applied after the running standard-scaler
loss_reward: two-hot symlog, 255 bins, linspace(-20, 20)
loss_reward_impl: reward_head dist=symlog_disc -> DiscDist (world_models/distributions.py:14)
loss_cont: Bernoulli on logits (cont_head dist=binary)
loss_scales: reward 1.0, cont 1.0, decoder 1.0
kl_free: 1.0
kl_dyn_scale: 0.5
kl_rep_scale: 0.1
grad_heads: [reward, cont, decoder]
cont_head_init_outscale: 0.0 (the code used reward_head.outscale; the config's cont_head.outscale 1.0 was never read)

## Architecture (all four final models)

dyn_stoch: 32
dyn_deter: 1024
dyn_hidden: 1024
dyn_discrete: 32  (WM2 / env3 / L2 only: 64)
units: 512
act: SiLU
norm: True
cnn_depth: 16
cnn_kernel_size: 4
cnn_minres: 4
mlp_layers: 2
mlp_units: 512

## SSL pretraining

ssl_epochs: 200
ssl_iterations_per_epoch: 200
ssl_gradient_steps: 40000
ssl_gradient_steps_source: 200 x 200, wtrain() in ssl_wms/src/trainers/ssl.py:163
ssl_batch: 128 x 64 (batch_sz x batch_length)
ssl_sample_frames_with_reuse: 327,680,000
ssl_eval_iterations_per_epoch: 200 (no grad)
ssl_optimizer: AdamW, lr 1e-4, eps 1e-8, weight_decay 0.01, grad_clip 1000
ssl_scheduler: none
ssl_checkpoints:
  WM1 (L1): ssl_wms/outputs/iros26/env2/2026-03-12/03-45-45  seed 60
  WM2 (L2): ssl_wms/outputs/iros26/env3/2026-04-07/14-33-56  seed 20
  WM3 (L3): ssl_wms/outputs/iros26/env4/2026-04-12/06-43-25  seed 20
  WM4 (L4): ssl_wms/outputs/iros26/env6/2026-03-21/05-08-41  seed 20
ssl_checkpoints_note: the fine-tuning configs point at outputs/sweep_2|env_3|env_4|env_6/...; those dirs were relocated to outputs/iros26/env{2,3,4,6}/ with the timestamps unchanged
ssl_checkpoint_is_last_epoch: agent_eval_best.pth (the file fine-tuning loaded) holds the LAST epoch's weights: TrainerSSL never updated previous_m_eval / previous_m_train, so the "best" files were overwritten every epoch. The architecture sweep picked runs by their minimum validation loss but used their last-epoch weights.
ssl_data_seed: 0 in every run (the buffers' RNG); the run seed (20/40/60) only set the model initialisation
ssl_code_state: config_ssl_wms.yaml at ssl_wms b77356a (2026-03-09) plus each run's recorded Hydra overrides reproduces its recorded config exactly, except train/valid_folder_list for WM2-4 (edited in the file, not overridden). The runs recorded their git hash/diff only in W&B, whose local files are gone.
ssl_hardware: 128 x 64 does not fit on a 12 GB GPU (legacy code included)

## SSL dataset

ssl_train_set_episodes: UNVERIFIED - not on disk
ssl_train_set_frames: UNVERIFIED - not on disk
ssl_train_set_note: each SSL run read path_to_data <ssl_wms>/datasets/ + train_folder_list ["train_env_N"] (which overrides folder_rule) + file_rule "*.pt", per the runs' .hydra/config.yaml. That directory no longer exists. Collection used num_envs 64 and eps_train 10000, which SUGGESTS ~10k episodes / ~700k frames at the measured ~70-frame mean, but that is inferred from the collection config, not measured. Do not quote as measured.
ssl_file_rule: "*.pt"  (as recorded by the runs; the repo's later "*_low_*.pt" belongs to the cross-validation pass)
ssl_validation_set: eval_env_N, same directory

### Surviving eval sets - datasets_data/iros26/datasets_new/ (measured 2026-08-12)

                          GiB  episodes    frames  mean  med  min  max  seq@64  hours
eval_env_2_low_freq.pt    5.0     3,061   215,907  70.5   65    2  181   3,373    6.0
eval_env_3_low_freq.pt    4.8     3,061   208,936  68.3   64    2  176   3,264    5.8
eval_env_4_low_freq.pt    4.8     3,062   207,513  67.8   63    2  200   3,242    5.8
eval_env_6_low_freq.pt    5.0     3,059   217,000  70.9   68    2  205   3,390    6.0
eval_env_2_high_freq.pt   3.1     3,061   135,446  44.2   40    2  198   2,116    3.8
eval_env_6_high_freq.pt   3.3     3,060   144,705  47.3   41    2  184   2,260    4.0

low_freq_total: 12,243 episodes | 849,356 frames | 23.6 h at 10 Hz
datasets_new_disk: 27 GiB
seq@64_note: frames // batch_length, i.e. the number of 64-step training sequences the loader carves out
hours_note: frames x 0.1 s
high_freq_note: exists for env2 and env6 only in datasets_new; whether the lost train_env_N folders contained high_freq files is unknown

episode_filtering: effectively none. The as-run path is wtrain=True, which samples with ReplayBuffer.sample() (ssl_wms/src/datasets/replay_buffer.py): it skips only trajectories with fewer than 2 steps and stitches segments of several episodes (marking is_first) until a batch_length sequence is filled. Short episodes are not discarded. (The WMDataset concatenate-and-chunk path is wtrain=False and was not used.)
min_length_50: applies at DATA COLLECTION (main_collect_dataset.py passes min_length to its buffer; [wmnav.collect min_length]): finished episodes of <= 50 steps (5 s) are dropped and not counted. The fine-tuning buffer is built without it. Verified on the eval sets: 3000 finished episodes each, only 3 of them <= 50 steps; the ~61 short ones are episodes still running when collection stopped.

dataset_generations: three, none containing the train_env_N folders the SSL runs read
  datasets/            2026-01-30..02-09  one envs_eval_N.pt per level, envs 2-6
  datasets_long_trjs/  2026-02-20         one envs_eval_N.pt per level, envs 2-6
  datasets_new/        2026-03-18 (env2, env6), 2026-04-09 (env3, env4)
generation_note: the datasets_new files postdate the env2 SSL sweep (2026-03-05..03-13) and the env6 checkpoint (2026-03-21 uses data written 03-18), so datasets_new is a later generation and is not what the March SSL runs read.

exploration_policy: safe_explore - corridor-centring controller on the depth image (world_models/agents.py:62)
exploration_policy_note: low_freq = safe_explore; high_freq = uniform/OU/Lissajous/spline mix (now commented out). Both recorded at 10 Hz - the names refer to action bandwidth, not sampling rate. Which of the two the lost SSL training sets contained is not recoverable from disk.

## RL fine-tuning

rl_seed: 5
rl_seed_source: configs/cfgs/static_env_wms.yaml:3, confirmed resolved to 5 in all four wandb configs
rl_deterministic_run: True in all four runs (resolved W&B configs; the repo file says False). It selects deterministic cuDNN kernels but training still was not reproducible - see rl_twin_runs.
rl_runs_per_model: 1 used in the paper; each model has a second run with the same seed (rl_twin_runs)
rl_twin_runs: fine_tuned_env3/2026-04-13/11-51-02, fine_tuned_env4/2026-04-15/11-20-12, fine_tuned_env6/2026-04-17/10-40-11 differ from the paper runs (other weights, other mean episode length); the env2 pair (2026-03-25 / 2026-03-29) is bit-identical. Never evaluated on L1-L4.
rl_compile: compile: True in the configs had no effect: torch.compile wrapped forward(), and the models were driven through other methods. All runs executed eagerly; the only trace is the _orig_mod. prefix in checkpoint keys.
rl_num_envs: 64
rl_prefill: 2500 frames (~39 env steps x 64 envs), random actions
rl_eps_train: 10000 episodes
rl_train_ratio: 512
rl_model_lr: 1e-7
rl_model_lr_note: config_wms.yaml in the repo now says 1e-5; the runs used 1e-7
rl_actor_lr: 3e-5
rl_critic_lr: 3e-5
rl_actor_entropy: 1e-3
rl_pretrain_steps: 100
rl_loaded_weights: world model only (_load_wm); actor and critic are trained from scratch

                episodes | mean_ep_len | env_frames | batch    | grad_steps
WM1 (env2, L1):   10,188 |        97.3 |    991,632 | 128 x 64 |   ~61,977
WM2 (env3, L2):   10,190 |        89.6 |    913,181 |  64 x 64 |  ~114,148
WM3 (env4, L3):   10,176 |        77.2 |    785,274 |  64 x 64 |   ~98,159
WM4 (env6, L4):   10,173 |        95.4 |    970,817 |  64 x 64 |  ~121,352

rl_frames_source: wandb-summary.json, log/n_episodes x metrics/step_env
rl_grad_steps_source: should_train = Every(batch_sz * batch_length / train_ratio), world_models/agents.py:180 [wmnav/dreamer/agent.py]
rl_wall_clock: 29-39 h per run
rl_checkpoints: outputs/iros26/new_dataset/fine_tuned_env{2,3,4,6}/.../net/best_train_win_rate.pt
rl_checkpoint_rule: saved whenever the mean win rate of the last 64 training episodes (incl. the random-action prefill episodes at the start) reaches a new maximum [wmnav/data/rollout.py BestTracker]
rl_hardware: host marvin-Lambda-Vector (W&B metadata); the 64 x 64 batch does not fit next to the simulator on a 12 GB GPU

## Seeds

ssl_seeds_on_disk: [20, 40, 60]  -> 3 distinct seeds
ssl_seeds_note: DISAGREES WITH APPENDIX if the appendix says 5. Across all 128 SSL runs under ssl_wms/outputs/iros26/env{2,3,4,6}/ the only seeds are 20, 40 and 60, and the seed variation was run only for the final architecture (batch_length 64, dyn_deter 1024, dyn_hidden 1024); the whole architecture grid is seed=20 only. sweep_cfg.yaml sweeps batch_length x dyn_deter x dyn_hidden and has no seed axis.
rl_seeds: [5] -> 1 seed, separate mechanism and value from the SSL seeds (twin runs with the same seed exist, rl_twin_runs)
env1_seeds: the dropped env1 model was fine-tuned with seeds 5, 10, 15 (moving-average win 0.984 / 0.953 / 0.969) - the only seed-variance evidence on disk
ppo_seeds: 1 per level, drawn at random (seed: null), stored in each experiment's config.json; SEED=n ./scripts/reproduce/ppo_levels.sh [wmnav] runs an independent sweep

## Controller randomization

controller: lmf2_velocity_control
controller_config: aerial_gym/config/controller_config/lmf2_controller_config.py
controller_config_status: UNMODIFIED standard AerialGym - aerial_gym_dev clean at dcd757dd, empty diff on this file
randomize_params: True
randomization_schedule: resampled per episode reset, uniform per env between min and max (base_lee_controller.py:87-108)

K_vel_min:    [2.7, 2.7, 1.7]
K_vel_max:    [3.3, 3.3, 1.3]
K_rot_min:    [1.6, 1.6, 0.25]
K_rot_max:    [1.85, 1.85, 0.4]
K_angvel_min: [0.4, 0.4, 0.075]
K_angvel_max: [0.5, 0.5, 0.09]
K_pos: [2.0, 2.0, 1.0] fixed - used by lee_position_control only, not by the velocity controller
K_vel_z_note: min 1.7 > max 1.3. torch_rand_float_tensor computes (max-min)*rand+min, so the sampled interval is still [1.3, 1.7]; harmless but reads as a typo.

max_inclination_angle_rad: pi/3
max_yaw_rate_rad_s: pi/3

motor_thrust_constant: [9.26312e-6, 1.826312e-5]
motor_time_constant_increasing_s: [0.05, 0.08]
motor_time_constant_decreasing_s: 0.005 (fixed)
motor_max_thrust_N: 10.0
motor_min_thrust_N: 0.1
thrust_to_torque_ratio: 0.07

force_torque_disturbance: DISABLED (LMF2Cfg.disturbance.enable_disturbance = False)
aerodynamic_damping: DISABLED (all damping coefficients 0.0)

## Evaluation sample sizes

wm_headline_table: games 237 / 259 / 295 / 238 (WM1-4 on their own level, 5 panels) - an accident of the debug-buffer flush, not a chosen n; win 98.7 / 91.5 / 93.9 / 88.2 %
may_cross_validation: outputs/iros26/cross_validation/2026-05-25, 4 models x 5 conditions x 200 games (16 envs, seed 5, video_pred_log on): L1, L2, Sobol (= L3: Sobol ignores base_list), L4, L4 with 10 panels
august_cross_eval_7_panels: 4 x 4 x 200 games
ppo_eval: ~1000 episodes per cell (948-1172), deterministic, latest checkpoint
standard_error: sqrt(p(1-p)/n): ~2.5 points at n = 200, ~1 point at n = 1000
reproduction: wmnav.evaluate on WM3 / L3 / 5 panels / 16 envs / seed 5 / 200 episodes = 187 wins, 13 crashes, identical to the legacy evaluator with video prediction off (the May run had it on, which consumes random numbers: 185 wins)
