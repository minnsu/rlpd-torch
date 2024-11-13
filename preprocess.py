# built-in modules
import h5py

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
    raise NotImplementedError


if __name__ == '__main__':
    preprocess('./dataset/raw/', './dataset/preprocessed/', _preprocess)