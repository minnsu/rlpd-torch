# built-in modules
import os

# 3rd-party modules
import torch
import numpy as np


# self-defined modules


def env_obs_to_model_obs(env_obs, device):
    """
    env_obs format:

    model_obs format:
    {
        "images": torch.DoubleTensor()
        "states": torch.Tensor()
    }
    """

    obs_img = None
    obs_state = None

    return {
        "images": obs_img.to(device),
        "states": obs_state.to(device)
    }

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
    epi_limit = 3
    preprocessed_data = []

    episodes = os.listdir(raw_path)
    episodes.sort()
    
    for idx, epi_n in enumerate(episodes):
        if idx >= epi_limit:
            break
        epi_path = os.path.join(raw_path, epi_n)
        
        print(f"============= {epi_n} ============")

        epi_data = {
            "images": torch.tensor([], dtype=torch.uint8),
            "states": torch.Tensor(),
            "actions": torch.Tensor(),
            "rewards": torch.Tensor(),
            "dones": torch.Tensor()
        }

        # Do preprocessing for your raw dataset
        # and set it to the return format

        preprocessed_data.append(epi_data)        

    return preprocessed_data