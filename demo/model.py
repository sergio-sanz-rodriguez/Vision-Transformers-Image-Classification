import torch
from pathlib import Path
from vision_transformer import ViT
from torchvision.transforms import v2

def load_model(model: torch.nn.Module,
               model_weights_dir: str,
               model_weights_name: str):
               #hidden_units: int):

    """Loads a PyTorch model from a target directory.

    Args:
    model: A target PyTorch model to load.
    model_weights_dir: A directory where the model is located.
    model_weights_name: The name of the model to load.
      Should include either ".pth" or ".pt" as the file extension.

    Example usage:
    model = load_model(model=model,
                       model_weights_dir="models",
                       model_weights_name="05_going_modular_tingvgg_model.pth")

    Returns:
    The loaded PyTorch model.
    """
    # Create the model directory path
    model_dir_path = Path(model_weights_dir)

    # Create the model path
    assert model_weights_name.endswith(".pth") or model_weights_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_path = model_dir_path / model_weights_name

    # Load the model
    print(f"[INFO] Loading model from: {model_path}")
    
    model.load_state_dict(torch.load(model_path, weights_only=True))
    
    return model

def create_vitbase_model(
    model_weights_dir:Path,
    model_weights_name:str,
    img_size:int=224,
    num_classes:int=101
    ):
    """
    Creates a ViT-B/16 model with the specified number of classes.

    Args:
    img_size: The size of the input image.
    num_classes: The number of classes for the classification task.

    Returns:
    The created ViT-B/16 model.
    """    
    # Instantiate the model
    vitbase16_model = ViT(
        img_size=img_size,
        in_channels=3,
        patch_size=16,
        num_transformer_layers=12,
        emb_dim=768,
        mlp_size=3072,
        num_heads=12,
        attn_dropout=0,
        mlp_dropout=0.1,
        emb_dropout=0.1,
        num_classes=num_classes
    )
    
    # Load the trained weights
    vitbase16_model = load_model(
        model=vitbase16_model,
        model_weights_dir=model_weights_dir,
        model_weights_name=model_weights_name
        )
    
    return vitbase16_model
