import torch
import torchvision
from pathlib import Path
from vision_transformer import ViT

def load_model(model: torch.nn.Module,
               model_weights_dir: str,
               model_weights_name: str):

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
    
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
    
    return model

def create_vitbase_model(
    model_weights_dir:Path,
    model_weights_name:str,
    img_size:int=224,
    num_classes:int=101,
    compile:bool=False
    ):
    """
    Creates a ViT-B/16 model with the specified number of classes.

    Args:
        model_weights_dir: A directory where the model is located.
        model_weights_name: The name of the model to load.
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
    
    # Compile the model
    if compile:
        vitbase16_model = torch.compile(vitbase16_model, backend="aot_eager")

    # Load the trained weights
    vitbase16_model = load_model(
        model=vitbase16_model,
        model_weights_dir=model_weights_dir,
        model_weights_name=model_weights_name
        )
    
    return vitbase16_model

# Create an EfficientNet-B0 Model
def create_effnetb0(
        model_weights_dir: Path,
        model_weights_name: str,
        num_classes: int=2,
        dropout: float=0.2,
        compile:bool=False
        ):
    """Creates an EfficientNetB0 feature extractor model and transforms.

    Args:
        model_weights_dir: A directory where the model is located.
        model_weights_name: The name of the model to load.
        num_classes (int, optional): number of classes in the classifier head.
        dropout (float, optional): Dropout rate. Defaults to 0.2.

    Returns:
        effnetb0_model (torch.nn.Module): EffNetB0 feature extractor model.
        transforms (torchvision.transforms): Image transforms.
    """
    
    # Load pretrained weights
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT # .DEFAULT = best available weights 
    effnetb0_model = torchvision.models.efficientnet_b0(weights=weights).to('cpu')

    # Recreate the classifier layer and seed it to the target device
    effnetb0_model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=dropout, inplace=True), 
        torch.nn.Linear(in_features=1280, 
                        out_features=num_classes,
                        bias=True))
    
    # Compile the model
    if compile:
        effnetb0_model = torch.compile(effnetb0_model, backend="aot_eager")
    
    # Create the model directory path
    model_dir_path = Path(model_weights_dir)

    # Create the model path
    assert model_weights_name.endswith(".pth") or model_weights_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_path = model_dir_path / model_weights_name

    # Load the state dictionary into the model
    effnetb0_model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
        
    return effnetb0_model
