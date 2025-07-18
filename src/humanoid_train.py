import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import types, sys, numpy as np
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
import os
import glob
import sys
import numpy
# tensorboard --logdir=./tensorboard_logs

def load_latest_checkpoint(path, env, device):
    checkpoints = glob.glob(os.path.join(path, "model_*.zip"))

    if not checkpoints:
        return None, 0

    # Extract (filepath, step) tuples
    def extract_step(filename):
        basename = os.path.splitext(os.path.basename(filename))[0]
        return int(basename.split("_")[-1])

    checkpoints = sorted(checkpoints, key=extract_step)
    latest = checkpoints[-1]
    step = extract_step(latest)

    print(f"Resuming from checkpoint: {latest} (step {step})")
    return SAC.load(latest, env=env, device=device), step

# To save current model for rendering progress
class CheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self):
        if self.num_timesteps % self.save_freq == 0:
            checkpoint_file = os.path.join(self.save_path, f"model_{self.num_timesteps}")
            self.model.save(checkpoint_file)
            if self.verbose > 0:
                print(f"Saved checkpoint at {checkpoint_file}")
        return True

class TensorboardRewardLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep_rew = info["episode"]["r"]
                self.episode_rewards.append(ep_rew)
                self.logger.record("rollout/ep_rew_mean", np.mean(self.episode_rewards[-100:]))
        return True

class TQDMCallback(BaseCallback):
    def __init__(self, total_timesteps, start_step=0, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.start_step = start_step
        self.pbar = None

    def _on_training_start(self):
        self.pbar = tqdm(
            total=self.total_timesteps,
            initial=self.start_step,
            desc="Training Progress"
        )

    def _on_step(self):
        self.pbar.update(self.model.n_envs)
        return True

    def _on_training_end(self):
        self.pbar.close()

class ActionRepeatWrapper(gym.Wrapper):
    def __init__(self, env, repeat=4):
        super().__init__(env)
        self.repeat = repeat

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class Float32ObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Convert observation space to float32
        self.observation_space = gym.spaces.Box(
            low=self.observation_space.low.astype(np.float32),
            high=self.observation_space.high.astype(np.float32),
            dtype=np.float32,
        )

    def observation(self, obs):
        return obs.astype(np.float32)

# Choose environment
def make_env(seed=42):
   
    '''
    | Parameter                                    | Type      | Default          | Description                                                                                                                                                                                                 |
    | -------------------------------------------- | --------- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | `xml_file`                                   | **str**   | `"humanoid.xml"` | Path to a MuJoCo model                                                                                                                                                                                      |
    | `forward_reward_weight`                      | **float** | `1.25`           | Weight for _forward_reward_ term (see `Rewards` section)                                                                                                                                                    |
    | `ctrl_cost_weight`                           | **float** | `0.1`            | Weight for _ctrl_cost_ term (see `Rewards` section)                                                                                                                                                         |
    | `contact_cost_weight`                        | **float** | `5e-7`           | Weight for _contact_cost_ term (see `Rewards` section)                                                                                                                                                      |
    | `contact_cost_range`                         | **float** | `(-np.inf, 10.0)`| Clamps the _contact_cost_ term (see `Rewards` section)                                                                                                                                                      |
    | `healthy_reward`                             | **float** | `5.0`            | Weight for _healthy_reward_ term (see `Rewards` section)                                                                                                                                                    |
    | `terminate_when_unhealthy`                   | **bool**  | `True`           | If `True`, issue a `terminated` signal is unhealthy (see `Episode End` section)                                                                                                                                |
    | `healthy_z_range`                            | **tuple** | `(1.0, 2.0)`     | The humanoid is considered healthy if the z-coordinate of the torso is in this range (see `Episode End` section)                                                                                            |
    | `reset_noise_scale`                          | **float** | `1e-2`           | Scale of random perturbations of initial position and velocity (see `Starting State` section)                                                                                                               |
    | `exclude_current_positions_from_observation` | **bool**  | `True`           | Whether or not to omit the x- and y-coordinates from observations. Excluding the position can serve as an inductive bias to induce position-agnostic behavior in policies (see `Observation State` section) |
    | `include_cinert_in_observation`              | **bool**  | `True`           | Whether to include *cinert* elements in the observations (see `Observation State` section)                                                                                                                  |
    | `include_cvel_in_observation`                | **bool**  | `True`           | Whether to include *cvel* elements in the observations (see `Observation State` section)                                                                                                                    |
    | `include_qfrc_actuator_in_observation`       | **bool**  | `True`           | Whether to include *qfrc_actuator* elements in the observations (see `Observation State` section)                                                                                                           |
    | `include_cfrc_ext_in_observation`            | **bool**  | `True`           | Whether to include *cfrc_ext* elements in the observations (see `Observation State` section)                                                                                                                |
    '''
   
    xml_file = "../models/humanoid.xml"
    #   xml_file="../models/custom_humanoid_2xSluggish.xml"
    env = gym.make("Humanoid-v5",
                    xml_file = xml_file,
                    forward_reward_weight=3,
                    ctrl_cost_weight=0.05,
                    contact_cost_weight=1e-7,
                    healthy_reward=5.0,
                    healthy_z_range= (0.65,2.0),
                    exclude_current_positions_from_observation=True
                    )
    
    env = Float32ObsWrapper(env)  # convert to float32
    env = Monitor(env) 
    env.reset(seed=seed)
    return env
    
def main(num_steps = 20_000_000, test_name = 'RunningTest_DefaultXML'):
    
    checkpoint_path = '../data/checkpoints/checkpoints_' + test_name
    

    # SB3 requires a vectorized environment
    env = DummyVecEnv([make_env])
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    

    model, start_step = load_latest_checkpoint(checkpoint_path, env, device)
    if model is None:
        print("Starting fresh training...")
        model = SAC("MlpPolicy", env, verbose=0, device=device)
        start_step = 0
    else:
        print(f"Resuming training from {start_step:,} steps...")

    # Set logger to TensorBoard path
    log_path = "../data/tensorboard_logs/SAC_Humanoid_" + test_name
    new_logger = configure(log_path, ["tensorboard"])
    model.set_logger(new_logger)

    # Update callbacks
    total_steps = start_step + num_steps
    callback = [
        TQDMCallback(total_timesteps=total_steps, start_step=start_step),
        CheckpointCallback(save_freq=int(num_steps / 20), save_path=checkpoint_path, verbose=1),
        TensorboardRewardLogger(),
    ]

    # Train step
    model.learn(
        total_timesteps=total_steps,
        reset_num_timesteps=False,
        callback=callback
    )

    model.save("humanoid_learned_" + test_name)


if __name__ == "__main__":
    # tensorboard --logdir=./tensorboard_logs
    main()