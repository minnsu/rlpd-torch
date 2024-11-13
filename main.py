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
    env = gym.make('CartPole-v1')
    env.seed(config['seed'])

    # =============== [MODEL & AGENT] ===============
    model = Model(state_dim=4, action_dim=1, output_dim=1).to(device)
    agent = SACAgent(model=model)

    # ================= [TRAINING] ==================
    obs, done = env.reset(), False
    
    # [DEBUGGING] -------------------------------
    img = env.render(mode='rgb_array')
    obs = {
        'images': torch.FloatTensor(img.transpose(2, 0, 1)).unsqueeze(0).to(device),
        'states': torch.FloatTensor(obs).to(device),
    }
    prev_ts = 0
    # --------------------------------------------

    for ts in range(config['max_steps']):
        action = agent(obs)

        # [DEBUGGING] --------------------------------
        action = 1 if action > 0 else 0
        # --------------------------------------------

        next_obs, reward, done, _ = env.step(action)
        # [DEBUGGING] --------------------------------
        img = env.render(mode='rgb_array')
        next_obs = {
            'images': torch.FloatTensor(img.transpose(2, 0, 1)).unsqueeze(0).to(device),
            'states': torch.FloatTensor(next_obs).to(device),
        }
        # --------------------------------------------
        buffer_idx = add_to_replay_buffer(replay_buffer, buffer_idx, obs, action, reward, next_obs, done)
        obs = next_obs

        if ts > config["start_training_after"] and ts % config['update_period'] == 0:
            offline_batch = sample_batch(offline_data, config['batch_size'] // 2)
            online_batch = sample_batch(replay_buffer, config['batch_size'] // 2)

            batch = combine_batch(offline_batch, online_batch)
            batch = batch_to_gpu(batch, device)

            agent.update(batch)

        if ts == config['ckpt_period']:
            torch.save(agent.model.state_dict(), os.path.join(config['ckpt_dir'], f"model_{ts}.pth"))
            # [DEBUGGING] -------------------------------
            sim_evaluation(env, agent, device)
            # -------------------------------------------


        if done:
            obs, done = env.reset(), False
            # [DEBUGGING] -------------------------------
            print(f"Episode finished. epi steps: {ts - prev_ts}")
            img = env.render(mode='rgb_array')
            obs = {
                'images': torch.FloatTensor(img.transpose(2, 0, 1)).unsqueeze(0).to(device),
                'states': torch.FloatTensor(obs).to(device),
            }
            prev_ts = ts
            # --------------------------------------------

if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    main(config)