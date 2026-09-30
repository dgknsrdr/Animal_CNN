import torch
from pathlib import Path


def save_model(model : torch.nn.Module,
               target_dir : str,
               model_name : str,
               class_names):

    target_dir_path = Path(target_dir)

    target_dir_path.mkdir(
        parents=True,
        exist_ok=True)

    model_save_path =target_dir_path / model_name

    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names
    }, "../models/efficientnet_animals.pth")