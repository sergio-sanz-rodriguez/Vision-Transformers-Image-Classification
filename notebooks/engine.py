"""
Contains functions for training and testing a PyTorch model.
"""

import os
import logging
import torch
import torchvision
import random
import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
from tqdm.auto import tqdm 
from torch.utils.tensorboard import SummaryWriter
#from typing import Optional
#from torchvision import transforms
#from torchvision.transforms import v2
from IPython.display import clear_output
import matplotlib.pyplot as plt
from pathlib import Path
from timeit import default_timer as timer
from PIL import Image
from torch import GradScaler, autocast
from sklearn.metrics import precision_recall_curve, classification_report


def sec_to_min_sec(seconds):
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    return f"{int(minutes)}m{int(remaining_seconds)}s"
    
# Calculate accuracy (a classification metric)
def calculate_accuracy(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def calculate_fpr_at_recall(y_true, y_pred_probs, recall_threshold):
    """Calculates the False Positive Rate (FPR) at a specified recall threshold.

    Args:
        y_true (torch.Tensor): Ground truth labels.
        y_pred_probs (torch.Tensor): Predicted probabilities.
        recall_threshold (float): The recall threshold at which to calculate the FPR.

    Returns:
        float: The calculated FPR at the specified recall threshold.
    """

    if not (0 <= recall_threshold <= 1):
        raise ValueError("recall_threshold must be between 0 and 1.")

    # Convert list to tensor if necessary
    if isinstance(y_pred_probs, list):
        y_pred_probs = torch.cat(y_pred_probs)  
    if isinstance(y_true, list):
        y_true = torch.cat(y_true)

    n_classes = y_pred_probs.shape[1]
    fpr_per_class = []

    for class_idx in range(n_classes):
        # Get true binary labels and predicted probabilities for the class
        y_true_bin = (y_true == class_idx).int().detach().numpy()
        y_scores = y_pred_probs[:, class_idx].detach().numpy()

        # Compute precision-recall curve
        _, recall, thresholds = precision_recall_curve(y_true_bin, y_scores)

        # Find the threshold closest to the desired recall
        idx = np.where(recall >= recall_threshold)[0]
        if len(idx) > 0:
            threshold = thresholds[idx[-1]]
            # Calculate false positive rate
            fp = np.sum((y_scores >= threshold) & (y_true_bin == 0))
            tn = np.sum((y_scores < threshold) & (y_true_bin == 0))
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        else:
            fpr = 0  # No threshold meets the recall condition

        fpr_per_class.append(fpr)

    return np.mean(fpr_per_class)  # Average FPR across all classes

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving best model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)

def load_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    
    """Loads a PyTorch model from a target directory.

    Args:
    model: A target PyTorch model to load.
    target_dir: A directory where the model is located.
    model_name: The name of the model to load.
      Should include either ".pth" or ".pt" as the file extension.

    Example usage:
    model = load_model(model=model,
                       target_dir="models",
                       model_name="model.pth")

    Returns:
    The loaded PyTorch model.
    """
    # Create the model directory path
    model_dir_path = Path(target_dir)

    # Create the model path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_path = model_dir_path / model_name

    # Load the model
    print(f"[INFO] Loading model from: {model_path}")
    
    model.load_state_dict(torch.load(model_path, weights_only=True))
    
    return model


def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               recall_threshold: float=None,
               amp: bool=True,
               device: torch.device = "cuda" if torch.cuda.is_available() else "cpu") -> Tuple[float, float]:
    
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    recall_threshold: The recall threshold at which to calculate the FPR (between 0 and 1)
    amp: Whether to use mixed precision training (True) or not (False)
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()
    model.to(device)

    # Initialize the GradScaler for  Automatic Mixed Precision (AMP)
    scaler = GradScaler() if amp else None

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0    
    all_preds = []
    all_labels = []

    # Loop through data loader data batches
    for _, (X, y) in tqdm(enumerate(dataloader), total=len(dataloader)): 
        #, desc=f"Training epoch {epoch_number}..."):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # Optimize training with amp if available
        if amp:
            with autocast(device_type='cuda', dtype=torch.float16):
                # Forward pass
                y_pred = model(X)
                y_pred = y_pred.contiguous()
                
                # Calculate  and accumulate loss
                loss = loss_fn(y_pred, y)
            
            # Optimizer zero grad
            optimizer.zero_grad()

            # Backward pass and optimizer step with scaled gradients
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            # Forward pass
            y_pred = model(X)
            y_pred = y_pred.contiguous()
            
            # Calculate  and accumulate loss
            loss = loss_fn(y_pred, y)

            # Optimizer zero grad
            optimizer.zero_grad()

            # Loss backward
            loss.backward()

            # Optimizer step
            optimizer.step()

        # Calculate and accumulate loss and accuracy across all batches
        train_loss += loss.item()
        #y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        y_pred_class = y_pred.argmax(dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

        if recall_threshold:
            all_preds.append(torch.softmax(y_pred, dim=1).detach().cpu())
            all_labels.append(y.detach().cpu())

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    # Final FPR calculation
    if recall_threshold and all_preds:
        try:
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)
            fpr_at_recall = calculate_fpr_at_recall(all_labels, all_preds, recall_threshold)
        except Exception as e:
            logging.error(f"Error calculating final FPR at recall: {e}")
            fpr_at_recall = None
    else:
        fpr_at_recall = None

    return train_loss, train_acc, fpr_at_recall

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              recall_threshold: float=None,
              device: torch.device = "cuda" if torch.cuda.is_available() else "cpu") -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval() 
    model.to(device)

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0
    all_preds = []
    all_labels = []

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for _, (X, y) in tqdm(enumerate(dataloader), total=len(dataloader), colour='#FF9E2C'):
            #, desc=f"Validating epoch {epoch_number}..."):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred = model(X)
            test_pred = test_pred.contiguous()

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_class = test_pred.argmax(dim=1)
            test_acc += ((test_pred_class == y).sum().item()/len(test_pred))

            if recall_threshold:
                all_preds.append(torch.softmax(test_pred, dim=1).detach().cpu())
                all_labels.append(y.detach().cpu())

    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)

    # Concatenate lists into tensors
    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)

    # Calculate FPR at specified recall threshold
    fpr_at_recall = calculate_fpr_at_recall(all_labels, all_preds, recall_threshold) if recall_threshold else None

    return test_loss, test_acc, fpr_at_recall

# Add writer parameter to train()
def train(model: torch.nn.Module,
          target_dir: str,
          model_name: str,
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          recall_threshold: float=None,
          scheduler: torch.optim.lr_scheduler=None, #Optional[torch.optim.lr_scheduler.LRScheduler]=None,
          epochs: int=30,
          device: torch.device="cuda" if torch.cuda.is_available() else "cpu", 
          plot_curves: bool=True,
          amp: bool=True,
          save_best_model: bool=True,
          writer: SummaryWriter=False
          ) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Stores metrics to specified writer log_dir if present.

    Args:
        model: A PyTorch model to be trained and tested.
        target_dir: A directory for saving the model to.
        model_name: A filename for the saved model. Should include
            either ".pth" or ".pt" as the file extension.
        train_dataloader: A DataLoader instance for the model to be trained on.
        test_dataloader: A DataLoader instance for the model to be tested on.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        loss_fn: A PyTorch loss function to calculate loss on both datasets.
        scheduler: A PyTorch learning rate scheduler to adjust the learning rate during training.
        epochs: An integer indicating how many epochs to train for.
        device: A target device to compute on (e.g. "cuda" or "cpu").
        plot: A boolean indicating whether to plot the training and testing curves.
        amp: A boolean indicating whether to use Automatic Mixed Precision (AMP) during training
        save_best_model: A boolean indicating whether to save the best model during training
        writer: A SummaryWriter() instance to log model results to.

    Returns:
        A dictionary of training and testing loss as well as training and
        testing accuracy metrics. Each metric has a value in a list for 
        each epoch.
        In the form: {
            train_loss: [...],
            train_acc: [...],
            test_loss: [...],
            test_acc: [...],
            train_time: [...],
            test_time: [...],
            lr: [...],
            train_fpr_at_recall: [...],
            test_fpr_at_recall: [...]
            } 
        For example if training for epochs=2: 
            {
            train_loss: [2.0616, 1.0537],
            train_acc: [0.3945, 0.3945],
            test_loss: [1.2641, 1.5706],
            test_acc: [0.3400, 0.2973],
            train_time: [1.1234, 1.5678],
            test_time: [0.4567, 0.7890],
            lr: [0.001, 0.0005],
            train_fpr_at_recall: [0.1234, 0.2345],
            test_fpr_at_recall: [0.3456, 0.4567]
            } 
    """

    # Define colors
    BLACK = '\033[30m'  # Black
    BLUE = '\033[34m'   # Blue
    ORANGE = '\033[38;5;214m'  # Orange (using extended colors)
    GREEN = '\033[32m'  # Green
    RESET = '\033[39m'  # Reset to default color

    # Create empty results dictionary
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "train_time [s]": [],
        "test_time [s]": [],
        "lr": [] 
        }
    if recall_threshold:
        results.update(
            {"train_fpr_at_recall": [],
             "test_fpr_at_recall": []
            }
        )

    # Initialize the best validation loss
    if save_best_model:
        model_name_best = model_name.replace(".", "_best.")
        best_test_loss = float("inf") 

    # Loop through training and testing steps for a number of epochs
    for epoch in range(epochs):

        # Perform training step and time it
        print(f"Training epoch {epoch+1}...")
        train_epoch_start_time = time.time()        
        train_loss, train_acc, train_fpr_at_recall = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            recall_threshold=recall_threshold,
            amp=amp,
            device=device)
        train_epoch_end_time = time.time()
        train_epoch_time = train_epoch_end_time - train_epoch_start_time

        # Perform test step and time it
        print(f"Validating epoch {epoch+1}...")
        test_epoch_start_time = time.time()
        test_loss, test_acc, test_fpr_at_recall = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            recall_threshold=recall_threshold,                             
            device=device)
        test_epoch_end_time = time.time()
        test_epoch_time = test_epoch_end_time - test_epoch_start_time

        clear_output(wait=True)

        # Print out what's happening
        if scheduler:
            lr = scheduler.get_last_lr()[0]
        else:
            lr = optimizer.param_groups[0]['lr']

        print(
            f"{BLACK}Epoch: {epoch+1} | "
            f"{BLUE}train_loss: {train_loss:.4f} {BLACK}| "
            f"{BLUE}train_acc: {train_acc:.4f} {BLACK}| "
            f"{BLUE}fpr_at_recall: {train_fpr_at_recall if recall_threshold else 0:.4f} {BLACK}| "
            f"{BLUE}train_time: {sec_to_min_sec(train_epoch_time)} {BLACK}| "            
            f"{ORANGE}test_loss: {test_loss:.4f} {BLACK}| "
            f"{ORANGE}test_acc: {test_acc:.4f} {BLACK}| "
            f"{ORANGE}fpr_at_recall: {test_fpr_at_recall if recall_threshold else 0:.4f} {BLACK}| "
            f"{ORANGE}test_time: {sec_to_min_sec(test_epoch_time)} {BLACK}| "
            f"{GREEN}lr: {lr:.10f}"
        )
    
        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        results["train_time [s]"].append(train_epoch_time)
        results["test_time [s]"].append(test_epoch_time)
        results["lr"].append(lr)
        if recall_threshold:
            results["train_fpr_at_recall"].append(train_fpr_at_recall)
            results["test_fpr_at_recall"].append(test_fpr_at_recall)

        # Scheduler step after the optimizer
        scheduler.step() if scheduler else None

        # Plot loss and accuracy curves
        n_plots = 3 if recall_threshold else 2
        if plot_curves:
            
            plt.figure(figsize=(20, 6))
            range_epochs = range(1, len(results["train_loss"])+1)

            # Plot loss
            plt.subplot(1, n_plots, 1)
            plt.plot(range_epochs, results["train_loss"], label="train_loss")
            plt.plot(range_epochs, results["test_loss"], label="test_loss")
            plt.title("Loss")
            plt.xlabel("Epochs")
            plt.grid(visible=True, which="both", axis="both")
            plt.legend()

            # Plot accuracy
            plt.subplot(1, n_plots, 2)
            plt.plot(range_epochs, results["train_acc"], label="train_accuracy")
            plt.plot(range_epochs, results["test_acc"], label="test_accuracy")
            plt.title("Accuracy")
            plt.xlabel("Epochs")
            plt.grid(visible=True, which="both", axis="both")
            plt.legend()
            
            # Plot FPR at recall
            if recall_threshold:
                plt.subplot(1, n_plots, 3)
                plt.plot(range_epochs, results["train_fpr_at_recall"], label="train_fpr_at_recall")
                plt.plot(range_epochs, results["test_fpr_at_recall"], label="test_fpr_at_recall")
                plt.title(f"FPR at {recall_threshold * 100}% recall")
                plt.xlabel("Epochs")
                plt.grid(visible=True, which="both", axis="both")
                plt.legend()
            
            plt.show()

        # Update the tqdm progress bar description with the current epoch
        #epoch_bar.set_description(f"Epoch {epoch+1}/{epochs}")
        #epoch_bar.update(1)

        # See if there's a writer, if so, log to it
        if writer:
            # Add results to SummaryWriter
            writer.add_scalars(main_tag="Loss", 
                               tag_scalar_dict={"train_loss": train_loss,
                                                "test_loss": test_loss},
                               global_step=epoch)
            writer.add_scalars(main_tag="Accuracy", 
                               tag_scalar_dict={"train_acc": train_acc,
                                                "test_acc": test_acc}, 
                               global_step=epoch)
            if recall_threshold:
                writer.add_scalars(main_tag="FPR at Recall", 
                                   tag_scalar_dict={"train_fpr_at_recall": train_fpr_at_recall,
                                                    "test_fpr_at_recall": test_fpr_at_recall}, 
                                   global_step=epoch)
        else:
            pass

        # Check if this is the best test loss so far and save the model
        if save_best_model and (test_loss < best_test_loss):
            best_test_loss = test_loss
            save_model(model=model,
                       target_dir=target_dir,
                       model_name=model_name_best)

    # Close the writer
    writer.close() if writer else None

    # Save the model (last epoch)
    save_model(model=model,
               target_dir=target_dir,
               model_name=model_name)

    # Return and save the filled results at the end of the epochs
    name , _ = model_name.rsplit('.', 1)
    csv_file_name = f"{name}.csv"
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(target_dir, csv_file_name), index=False)

    return df_results

def eval_model(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               device: torch.device = "cuda" if torch.cuda.is_available() else "cpu"):
    """Evaluates a given model on a given dataset.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        device (str, optional): Target device to compute on. Defaults to device.

    Returns:
        (dict): Results of model making predictions on data_loader.
    """
    loss, acc = 0, 0

    model.to(device)  #may not be necessary
    
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # Send data to the target device
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss += loss_fn(y_pred, y)
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            acc += ((y_pred_class == y).sum().item()/len(y_pred_class))            
        
        # Scale loss and acc
        loss /= len(data_loader)
        acc /= len(data_loader)

    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}


def make_predictions(model: torch.nn.Module,
                     dataloader: torch.utils.data.DataLoader,
                     device: torch.device = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    Predicts classes for a given dataset using a trained model.

    Args:
        model (torch.nn.Module): A trained PyTorch model.
        dataloader (torch.utils.data.DataLoader): The dataset to predict on.
        device (str, optional): Target device to compute on. Defaults to device.

    Returns:
        (list): All of the predicted class labels.
    """

    y_preds = []
    model.eval()
    model.to(device)
    with torch.inference_mode():
        for X, y in tqdm(dataloader, desc="Making predictions"):

            # Send data and targets to target device
            X, y = X.to(device), y.to(device)
            
            # Do the forward pass
            y_logit = model(X)

            # Turn predictions from logits -> prediction probabilities -> predictions labels
            y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)
            
            # Put predictions on CPU for evaluation
            y_preds.append(y_pred.cpu())

    # Concatenate list of predictions into a tensor
    return torch.cat(y_preds)


def pred_and_store(model: torch.nn.Module,
                   test_dir: str, 
                   transform: torchvision.transforms, 
                   class_names: List[str], 
                   percent_samples: float = 1.0,
                   seed=42,
                   device: str = "cuda" if torch.cuda.is_available() else "cpu") -> List[Dict]:
    
    """
    Returns a list of dictionaries with sample predictions, sample names, prediction probabilities, prediction times, actual labels and prediction classes.

    Args:
        paths: a list of target sample paths
    """
    # Create a list of test images and checkout existence
    print(f"[INFO] Finding all filepaths ending with '.jpg' in directory: {test_dir}")
    paths = list(Path(test_dir).glob("*/*.jpg"))
    assert len(list(paths)) > 0, f"No files ending with '.jpg' found in this directory: {test_dir}"

    # Number of random images to extract
    num_samples = len(paths)
    num_random_images = int(percent_samples * num_samples)

    # Ensure the number of images to extract is less than or equal to the total number of images
    assert num_random_images <= len(paths), f"Number of images to extract exceeds total images in directory: {len(paths)}"

    # Randomly select a subset of file paths
    torch.manual_seed(seed)
    paths = random.sample(paths, num_random_images)

    # Store predictions and ground-truth labels
    y_true = []
    y_pred = []

    # Create an empty list to store prediction dictionaires
    pred_list = []
    
    # Loop through target paths
    for path in tqdm(paths, total=num_samples):
        
        # Create empty dictionary to store prediction information for each sample
        pred_dict = {}

        # Get the sample path and ground truth class name
        pred_dict["image_path"] = path
        class_name = path.parent.stem
        pred_dict["class_name"] = class_name
        
        # Start the prediction timer
        start_time = timer()
        
        # Open image path
        img = Image.open(path)
        
        # Transform the image, add batch dimension and put image on target device
        transformed_image = transform(img).unsqueeze(0).to(device) 
        
        # Prepare model for inference by sending it to target device and turning on eval() mode
        model.to(device)
        model.eval()
        
        # Get prediction probability, predicition label and prediction class
        with torch.inference_mode():
            pred_logit = model(transformed_image) # perform inference on target sample 
            #pred_logit = pred_logit.contiguous()
            pred_prob = torch.softmax(pred_logit, dim=1) # turn logits into prediction probabilities
            pred_label = torch.argmax(pred_prob, dim=1) # turn prediction probabilities into prediction label
            pred_class = class_names[pred_label.cpu()] # hardcode prediction class to be on CPU

            # Make sure things in the dictionary are on CPU (required for inspecting predictions later on) 
            pred_dict["pred_prob"] = round(pred_prob.unsqueeze(0).max().cpu().item(), 4)
            pred_dict["pred_class"] = pred_class
            
            # End the timer and calculate time per pred
            end_time = timer()
            pred_dict["time_for_pred"] = round(end_time-start_time, 4)

        # Does the pred match the true label?
        pred_dict["correct"] = class_name == pred_class

        # Add the dictionary to the list of preds
        pred_list.append(pred_dict)

        # Append true and predicted label indexes
        y_true.append(class_names.index(class_name))
        y_pred.append(pred_label.cpu().item())

    # Generate the classification report
    classification_report_dict = classification_report(
        y_true=y_true,
        y_pred=y_pred,
        target_names=class_names,
        output_dict=True
    )

    # Return list of prediction dictionaries
    return pred_list, classification_report_dict


def create_writer(experiment_name: str, 
                  model_name: str, 
                  extra: str=None) -> torch.utils.tensorboard.writer.SummaryWriter():
    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

    Where timestamp is the current date in YYYY-MM-DD format.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        extra (str, optional): Anything extra to add to the directory. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.

    Example usage:
        # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
        writer = create_writer(experiment_name="data_10_percent",
                               model_name="effnetb2",
                               extra="5_epochs")
        # The above is the same as:
        writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
    """

    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now().strftime("%Y-%m-%d") # returns current date in YYYY-MM-DD format

    if extra:
        # Create log directory path
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)
        
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)


def train_step_v2(model: torch.nn.Module, 
                  dataloader: torch.utils.data.DataLoader, 
                  loss_fn: torch.nn.Module, 
                  optimizer: torch.optim.Optimizer,
                  recall_threshold: float=None,
                  amp: bool=True,
                  accumulation_steps: int = 1,
                  device: torch.device = "cuda" if torch.cuda.is_available() else "cpu") -> Tuple[float, float]:
    
    """Trains a PyTorch model for a single epoch with gradient accumulation.

    Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        loss_fn: A PyTorch loss function to minimize.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        amp: Whether to use mixed precision training (True) or not (False).
        device: A target device to compute on (e.g. "cuda" or "cpu").
        accumulation_steps: Number of mini-batches to accumulate gradients before an optimizer step.

    Returns:
        A tuple of training loss and training accuracy metrics.
        In the form (train_loss, train_accuracy). For example:

        (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()
    model.to(device)

    # Initialize the GradScaler for Automatic Mixed Precision (AMP)
    scaler = GradScaler() if amp else None

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0    
    all_preds = []
    all_labels = []

    # Loop through data loader data batches
    optimizer.zero_grad()  # Clear gradients before starting
    for batch, (X, y) in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # Optimize training with amp if available
        if amp:
            with autocast(device_type='cuda', dtype=torch.float16):
                # Forward pass
                y_pred = model(X)
                y_pred = y_pred.contiguous()
                
                # Calculate loss, normalize by accumulation steps
                loss = loss_fn(y_pred, y) / accumulation_steps
            
            # Backward pass with scaled gradients
            scaler.scale(loss).backward()

            # Gradient cliping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        else:
            # Forward pass
            y_pred = model(X)
            y_pred = y_pred.contiguous()
            
            # Calculate loss, normalize by accumulation steps
            loss = loss_fn(y_pred, y) / accumulation_steps

            # Backward pass
            loss.backward()

            # Gradient cliping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Perform optimizer step and clear gradients every accumulation_steps
        if (batch + 1) % accumulation_steps == 0 or (batch + 1) == len(dataloader):
            if amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()

        # Accumulate metrics
        train_loss += loss.item() * accumulation_steps  # Scale back to original loss
        y_pred_class = y_pred.argmax(dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

        if recall_threshold:
            all_preds.append(torch.softmax(y_pred, dim=1).detach().cpu())
            all_labels.append(y.detach().cpu())

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    # Final FPR calculation
    if recall_threshold and all_preds:
        try:
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)
            fpr_at_recall = calculate_fpr_at_recall(all_labels, all_preds, recall_threshold)
        except Exception as e:
            logging.error(f"Error calculating final FPR at recall: {e}")
            fpr_at_recall = None
    else:
        fpr_at_recall = None

    return train_loss, train_acc, fpr_at_recall
