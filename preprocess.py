# built-in modules
import os
import pickle as pkl
from PIL import Image

# 3rd-party modules
import torch
import numpy as np

# self-made modules
from utils import preprocess
from env_dependent_utils import _preprocess

if __name__ == '__main__':
    preprocess('../dataset/<raw_dataset>', '../dataset/<demo_data>.pkl', _preprocess)