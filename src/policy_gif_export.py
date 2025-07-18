import gymnasium as gym
import numpy as np
import mujoco
from mujoco import mj_step
from mujoco.glfw import MjRenderContextOffscreen

import time
import imageio
from stable_baselines3 import SAC

def main():
    # Load model
    model = SAC.load("../checkpoints/checkpoints_WalkingTest2xSluggish/model_14000000.zip", device="mps")

    # Load environment
    xml_file = "../models/custom_humanoid_2xSluggish.xml"
    env = gym.make("Humanoid-v5", xml_file=xml_file, healthy_z_range=(0.45, 2.0))
    obs, info = env.reset()

    model_obj = env.unwrapped.model
    data = env.unwrapped.data

    # Setup offscreen renderer
    ctx = MjRenderContextOffscreen(model_obj)
    ctx.make_current()

    frames = []

    for _ in range(250):
        action, _ = model.predict(obs.astype(np.float32), deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()

        mj_step(model_obj, data)
        ctx.render()

        rgb = ctx.read_pixels(640, 480, depth=False)[0]
        frames.append(np.flipud(rgb)) 

        time.sleep(1 / 60)

    imageio.mimsave("../data/FrameData/rendered_output.gif", frames, fps=30)

if __name__ == "__main__":
    main()
