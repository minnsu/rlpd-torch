# built-in modules
import os
import json

# 3rd-party modules
import torch
import numpy as np
import gym
import matplotlib.pyplot as plt

# self-made modules
from utils import load_data, convert_to_transition, default_replay_buffer
from utils import sample_batch, add_to_replay_buffer, combine_batch, batch_to_gpu
from utils import sim_evaluation
from agents import SACAgent
from models import Model

from env_dependent_utils import env_obs_to_model_obs


def main(config: dict):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # =================== [DATA] ====================
    offline_data = load_data(config['offline_path'])
    offline_data = convert_to_transition(offline_data)

    replay_buffer = default_replay_buffer(offline_data, config)
    buffer_idx = 0

    # ================ [ENVIRONMENT] ================
    env = None

    # =============== [MODEL & AGENT] ===============
    model = Model(state_dim=62, action_dim=16, output_dim=1).to(device)
    agent = SACAgent(model=model)

    # ================= [TRAINING] ==================
    obs, _ = env.reset()
    done = False
    
    # [DEBUGGING] -------------------------------
    obs = env_obs_to_model_obs(obs, device)
    train_idx = 0
    prev_ts = 0
    # --------------------------------------------

    mode = 'Acting'
    for ts in range(config['max_steps']):
        print(f"[{mode}] current timestep: {ts} / {config['max_steps']}", end='\r', flush=True)
        action = agent(obs)

        next_obs, reward, done, _, action = env.step(action.to('cpu'))
        
        # [DEBUGGING] --------------------------------
        next_obs = env_obs_to_model_obs(next_obs, device)
        action = torch.tensor(action).clone().detach().to(device)
        # --------------------------------------------

        buffer_idx = add_to_replay_buffer(replay_buffer, buffer_idx, obs, action, reward, next_obs, done)
        obs = next_obs

        if ts > config["start_training_after"] and ts % config['update_period'] == 0:
            mode = 'Training'
            print(f"[{mode}] current train idx: {train_idx}             ", end='\r', flush=True)

            offline_batch = sample_batch(offline_data, config['batch_size'] // 2)
            online_batch = sample_batch(replay_buffer, config['batch_size'] // 2)

            batch = combine_batch(offline_batch, online_batch)
            batch = batch_to_gpu(batch, device)

            agent.update(batch)

            train_idx += 1
            mode = 'Acting'

        if ts % config['ckpt_period'] == 0:
            torch.save(agent.model.state_dict(), os.path.join(config['ckpt_dir'], f"model_{ts}.pth"))

        if done:
            obs, _ = env.reset()
            done = False
            # [DEBUGGING] -------------------------------
            print(f"Episode finished. epi steps: {ts - prev_ts}")
            obs = env_obs_to_model_obs(obs, device)
            prev_ts = ts
            # --------------------------------------------

if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    main(config)