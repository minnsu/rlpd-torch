# built-in modules
import os
import json

# 3rd-party modules
import torch
import numpy as np
import gym

# self-made modules
from utils import load_data, convert_to_transition, default_replay_buffer
from utils import sample_batch, add_to_replay_buffer, combine_batch
from agents import SACAgent
from models import Model

def main(config: dict):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.seed(config['seed'])
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
    model = Model()
    agent = SACAgent(model=model)

    # ================= [TRAINING] ==================
    obs, done = env.reset(), False
    for ts in range(config['max_steps']):
        action = agent(obs)
        next_obs, reward, done, _ = env.step(action)

        buffer_idx = add_to_replay_buffer(replay_buffer, buffer_idx, obs, action, reward, next_obs, done)
        obs = next_obs

        if ts > config["start_training_after"]:
            offline_batch = sample_batch(offline_data, config['batch_size'] // 2)
            online_batch = sample_batch(replay_buffer, config['batch_size'] // 2)

            batch = combine_batch(offline_batch, online_batch)

            agent.update(batch)

        if ts == config['ckpt_period']:
            torch.save(agent.model.state_dict(), os.path.join(config['ckpt_dir'], f"model_{ts}.pth"))

        if done:
            obs, done = env.reset(), False

if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    main(config)