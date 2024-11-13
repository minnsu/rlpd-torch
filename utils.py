# built-in modules
import os
import pickle as pkl
from datetime import datetime

# 3rd-party modules
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

def preprocess(raw_path: str, data_path: str, _preprocess) -> dict:
    """
    Preprocess the raw data and store the preprocessed data in data_path.
    preprocessed_data (list[dict]): each dict corresponds to a episode of the raw data.
    [
        {
            'images': torch.Tensor, # The preprocessed images. shape: (N+1, C, H, W)
            'states': torch.Tensor, # The preprocessed states. shape: (N+1, D)
            'actions': torch.Tensor, # The actions. shape: (N, A)
            'rewards': torch.Tensor, # The rewards. shape: (N,)
            'dones': torch.Tensor, # The dones. shape: (N,)
        },
        ...
    ]

    Args:
        raw_path (str): The path to the raw data.
        data_path (str): The path to store the preprocessed data.

    Returns:
        None
    """
    # create the data_path if it does not exist
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    preprocessed_data = _preprocess(raw_path, data_path)
    
    # store the preprocessed data
    with open(os.path.join(data_path, f'{datetime.today()}.pkl'), 'wb') as f:
        pkl.dump(preprocessed_data, f)

    return None

def load_data(data_path: str) -> list[dict]:
    """
    Load the preprocessed data.

    Args:
        data_path (str): The path to the preprocessed data.

    Returns:
        data (list[dict]): each dict corresponds to a episode of the raw data.
    """
    with open(data_path, 'rb') as f:
        data = pkl.load(f)
    
    return data

def convert_to_transition(data: list[dict]) -> dict:
    """
    Convert the preprocessed data to transitions.

    Args:
        data (list[dict]): each dict corresponds to a episode of the raw data.
    
    Returns:
        transitions (dict): The transitions.
    """
    transitions = {
        'observations': {
            'images': torch.Tensor(),
            'states': torch.Tensor(),
        },
        'next_observations': {
            'images': torch.Tensor(),
            'states': torch.Tensor(),
        },
        'actions': torch.Tensor(),
        'rewards': torch.Tensor(),    
        'dones': torch.Tensor(),
    }

    for episode in data:
        transitions['observations']['images'] = torch.cat((transitions['observations']['images'],
                                                           episode['images'][:-1]), dim=0)
        transitions['observations']['states'] = torch.cat((transitions['observations']['states'],
                                                           episode['states'][:-1]), dim=0)
        
        transitions['next_observations']['images'] = torch.cat((transitions['next_observations']['images'],
                                                                episode['images'][1:]), dim=0)
        transitions['next_observations']['states'] = torch.cat((transitions['next_observations']['states'],
                                                                episode['states'][1:]), dim=0)
        
        transitions['actions'] = torch.cat((transitions['actions'], episode['actions']), dim=0)
        transitions['rewards'] = torch.cat((transitions['rewards'], episode['rewards']), dim=0)
        transitions['dones'] = torch.cat((transitions['dones'], episode['dones']), dim=0)

    return transitions

def sample_batch(data: dict, batch_size: int) -> dict:
    """
    Sample a batch of transitions from the preprocessed data.

    Args:
        data (dict): The preprocessed (transition) data.
        batch_size (int): The batch size.
    
    Returns:
        batch (dict): The batch of transitions.
    """
    indices = np.random.choice(len(data['observations']['states']), batch_size, replace=True)
    
    batch = {
        'observations': {
            'images': data['observations']['images'][indices],
            'states': data['observations']['states'][indices],
        },
        'next_observations': {
            'images': data['next_observations']['images'][indices],
            'states': data['next_observations']['states'][indices],
        },
        'actions': data['actions'][indices],
        'rewards': data['rewards'][indices],
        'dones': data['dones'][indices],
    }

    return batch

def combine_batch(demo_batch: dict, online_batch: dict) -> dict:
    """
    Combine the demo_batch and online_batch.

    Args:
        demo_batch (dict): The demo_batch.
        online_batch (dict): The online_batch.

    Returns:
        batch (dict): The combined batch.
    """
    batch = {
        'observations': {
            'images': torch.cat((demo_batch['observations']['images'], online_batch['observations']['images']), dim=0),
            'states': torch.cat((demo_batch['observations']['states'], online_batch['observations']['states']), dim=0),
        },
        'next_observations': {
            'images': torch.cat((demo_batch['next_observations']['images'], online_batch['next_observations']['images']), dim=0),
            'states': torch.cat((demo_batch['next_observations']['states'], online_batch['next_observations']['states']), dim=0),
        },
        'actions': torch.cat((demo_batch['actions'], online_batch['actions']), dim=0),
        'rewards': torch.cat((demo_batch['rewards'], online_batch['rewards']), dim=0),
        'dones': torch.cat((demo_batch['dones'], online_batch['dones']), dim=0),
    }

    return batch

buffer_size = None
def default_replay_buffer(demo_data: dict, config: dict) -> dict:
    """
    Get the default online data structure.

    Args:
        demo_data (dict): The preprocessed data.
        config (dict): The configuration.

    Returns:
        replay_buffer (dict): The default replay_buffer structure.
    """
    global buffer_size
    buffer_size = config['buffer_size']

    replay_buffer = {
        'observations': {
            'images': torch.zeros(config['buffer_size'], *demo_data['observations']['images'].shape[1:]),
            'states': torch.zeros(config['buffer_size'], *demo_data['observations']['states'].shape[1:]),
        },
        'next_observations': {
            'images': torch.zeros(config['buffer_size'], *demo_data['observations']['images'].shape[1:]),
            'states': torch.zeros(config['buffer_size'], *demo_data['observations']['states'].shape[1:]),
        },
        'actions': torch.zeros(config['buffer_size'], *demo_data['observations']['actions'].shape[1:]),
        'rewards': torch.zeros(config['buffer_size'], *demo_data['observations']['rewards'].shape[1:]),
        'dones': torch.zeros(config['buffer_size'], *demo_data['observations']['dones'].shape[1:]),
    }

    return replay_buffer

def add_to_replay_buffer(replay_buffer: dict, buffer_idx: int, obs, action, reward, next_obs, done) -> int:
    """
    Add the transition to the replay_buffer.

    Args:
        replay_buffer (dict): The replay_buffer.
    
    Returns:
        buffer_idx (int): The updated buffer_idx.
    """
    replay_buffer['observations']['images'][buffer_idx] = obs['images']
    replay_buffer['observations']['states'][buffer_idx] = obs['states']
    replay_buffer['actions'][buffer_idx] = action
    replay_buffer['rewards'][buffer_idx] = reward
    replay_buffer['next_observations']['images'][buffer_idx] = next_obs['images']
    replay_buffer['next_observations']['states'][buffer_idx] = next_obs['states']
    replay_buffer['dones'][buffer_idx] = done

    global buffer_size
    return (buffer_idx + 1) % buffer_size