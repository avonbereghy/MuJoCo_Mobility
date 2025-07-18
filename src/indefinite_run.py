import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer
from tqdm import tqdm
import time
class NoTerminateWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Override both
        return obs, reward, False, False, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

# Oscillation setup
# arm_joint_indices = [6, 7, 8, 9, 10, 11]
arm_joint_indices = [2]
frequency = 0.05
amplitude = 0.5

raw_env = gym.make("Humanoid-v5", render_mode=None, ctrl_cost_weight=0.1)
env = NoTerminateWrapper(raw_env)
obs, info = env.reset()

model = env.unwrapped.model
data = env.unwrapped.data
torso_id = model.body("torso").id

with mujoco.viewer.launch_passive(model, data) as viewer:
    base_action = np.zeros(env.action_space.shape[0])

    for step in tqdm(range(1000)):
        # Oscillating arm motion
        osc = amplitude * np.sin(2 * np.pi * frequency * step)
        action = base_action.copy()
        for idx in arm_joint_indices:
            action[idx] = osc

        obs, reward, terminated, truncated, info = env.step(action)

        torso_pos = data.body("torso").xpos
        viewer.cam.lookat[:] = torso_pos
        viewer.cam.distance = 5.0
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -20

        mujoco.mj_step(model, data)
        viewer.sync()

        # 60 FPS
        time.sleep(1 / 60)
