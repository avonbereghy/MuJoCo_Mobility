import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer
import time
from stable_baselines3 import SAC  # PPO, A2C
import glob
import json
import os


def main():

    data_folder = "checkpoints_WalkingTest2xSluggish"

    # Read all checkpoint paths and sort by number of training steps (tuples sort by first element)
    checkpoints = sorted([
        (int(path.split("/")[-1].split("_")[-1].replace(".zip", "")), path) 
        for path in glob.glob(f"../checkpoints/{data_folder}/model_*.zip")], 
        reverse=False)


    print(f"Loading...{checkpoints[-1]}")
    model = SAC.load(checkpoints[-1][1], device="mps")

    xml_file = "../models/custom_humanoid_2xSluggish.xml"
    
    # Load environment
    # Lower healthy z range show's falling behavior
    env = gym.make("Humanoid-v5",xml_file=xml_file, healthy_z_range= (0.45,2.0)) 

    obs, info = env.reset()
    model_obj = env.unwrapped.model
    data = env.unwrapped.data
    torso_id = model_obj.body("torso").id

    motion_log = []

    
    with mujoco.viewer.launch_passive(model_obj, data) as viewer:
        for step in range(250):
            # Convert to float32 for MPS
            action, _ = model.predict(obs.astype(np.float32), deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                obs, info = env.reset()

            # Camera
            torso_pos = data.body("torso").xpos
            viewer.cam.lookat[:] = torso_pos
            viewer.cam.distance = 5.0
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -20

            mujoco.mj_step(model_obj, data)
            viewer.sync()

            # Capture data per frame
            frame_data = {
                "qpos": data.qpos.copy().tolist(),
                "qvel": data.qvel.copy().tolist(),
                "body_xpos": data.xpos.copy().tolist(),  
                "body_xquat": data.xquat.copy().tolist(),
            }
            motion_log.append(frame_data)

            time.sleep(1 / 60)
    
    export_path = "../data/JointData/"
    os.makedirs(export_path, exist_ok=True)
    export_filename = os.path.join(export_path, "exported_motion_data.json")
    
    with open(export_filename, "w") as f:
        json.dump(motion_log, f)

    print(f"Joint motion data saved to '{export_filename}'")


if __name__ == "__main__":
    # mjpython policy_render.py
    # pip show mujoco
    main()



