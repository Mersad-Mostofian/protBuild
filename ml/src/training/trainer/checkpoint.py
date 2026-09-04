import os
from pathlib import Path
import torch

def save_model(model, checkpoint_path, save_file_name):
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    os.path.join(checkpoint_path, save_file_name)
    torch.save(model.state_dict(), checkpoint_path)