import os
from pathlib import Path
import torch

def save_model(checkpoint, checkpoint_path, save_file_name):
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_path, save_file_name)
    torch.save(checkpoint, checkpoint_path)