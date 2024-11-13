# built-in modules
import os
import pickle as pkl

# 3rd-party modules
import torch
import numpy as np

# self-made modules
from utils import preprocess

def _preprocess(raw_path: str, store_path: str) -> dict:
    """
    Define your own preprocess function here.

    Args:
        raw_path (str): The path to the raw data.
        store_path (str): The path to store the preprocessed data.
    
    Returns:
        preprocessed_data (list[dict]): each dict corresponds to a episode of the raw data.
        [
            {
                'images': torch.Tensor, # The preprocessed images. shape: (N, C, H, W)
                'states': torch.Tensor, # The preprocessed states. shape: (N, D)
                'actions': torch.Tensor, # The actions. shape: (N, A)
                'rewards': torch.Tensor, # The rewards. shape: (N,)
                'dones': torch.Tensor, # The dones. shape: (N,)
            },
            ...
        ] 
    """
    data = []
    for filename in os.listdir(raw_path):
        if filename.endswith('.pkl'):
            with open(os.path.join(raw_path, filename), 'rb') as f:
                data.append(pkl.load(f))

    preprocessed_data = []
    for epi in data:
        tmp = {
            'images': [],
            'states': [],
            'actions': [],
            'rewards': [],
            'dones': []
        }
        for trans in epi:
            # preprocess the images
            tmp['images'].append(trans['observations']['images'].transpose(2, 0, 1))
            tmp['states'].append(trans['observations']['states'])
            tmp['actions'].append(trans['actions'])
            tmp['rewards'].append(trans['rewards'])
            tmp['dones'].append(trans['dones'])
        
        for k, v in tmp.items():
            tmp[k] = torch.tensor(np.array(v))
        preprocessed_data.append(tmp)

    return preprocessed_data

if __name__ == '__main__':
    preprocess('./dataset/raw/', './dataset/preprocessed/cartpole-241113.pkl', _preprocess)