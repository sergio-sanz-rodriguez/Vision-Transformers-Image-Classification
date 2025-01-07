
# Imports and class names setup
import os
import torch
import json
import gradio as gr
from model import create_vitbase_model, create_effnetb0
from timeit import default_timer as timer
from typing import Tuple, Dict
from torchvision.transforms import v2


# Specify class names
food_vision_class_names_path = "class_names.txt"
with open(food_vision_class_names_path, "r") as f:
    class_names_102 = f.read().splitlines()
class_names_101 = class_names_102.copy()
class_names_101.remove("unknown")

# Specify number of classes
num_classes_102 = len(class_names_102) # 101 + unknown
num_classes_101 = len(class_names_101) # 101

# Load the food description file
food_descriptions_json = "food_descriptions.json"
with open(food_descriptions_json, 'r') as f:
    food_descriptions = json.load(f)

# Instantiate the model
classification_model_name_path = "effnetb0_2025-01-05_epoch13.pth"
effnetb0_model = create_effnetb0(
    model_weights_dir=".",
    model_weights_name=classification_model_name_path,
    num_classes=2,
    compile=True
    )

# Load the ViT-Base/16 transformer with input image of 384x384 pixels and 101 + unknown classes
vitbase_model_102 = create_vitbase_model(
    model_weights_dir=".",
    model_weights_name="vitbase16_102_2025-01-07.pth",
    img_size=384,
    num_classes=num_classes_102,
    compile=True
)

vitbase_model_101 = create_vitbase_model(
    model_weights_dir=".",
    model_weights_name="vitbase16_2_2024-12-31.pth",
    img_size=384,
    num_classes=num_classes_101,
    compile=True
)

# Specify manual transforms for model_2
transforms = v2.Compose([    
    v2.Resize((384)), #v2.Resize((384, 384)),
    v2.CenterCrop((384, 384)),    
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]) 
])


# Put models into evaluation mode and turn on inference mode
effnetb0_model.eval()
vitbase_model_102.eval()
vitbase_model_101.eval()

# Set thresdholds
BINARY_CLASSIF_THR = 0.9989122152328491
MULTICLASS_CLASSIF_THR = 0.5
ENTROPY_THR = 2.6

# Set model names
lite_model = "⚡ ViT Lite ⚡ faster, less accurate prediction"
pro_model =  "💎 ViT Pro 💎 slower, more accurate prediction"

# Predict method
def predict(image, model=pro_model) -> Tuple[Dict, str, str]:

    """Transforms and performs a prediction on image and returns prediction and time taken.
    """
    try:

        # Start the timer
        start_time = timer()

        # Transform the target image and add a batch dimension
        image = transforms(image).unsqueeze(0)
        
        # Make prediction...
        with torch.inference_mode():
            
            # If the picture is food
            if effnetb0_model(image)[:,1].cpu() >= BINARY_CLASSIF_THR:

                # If Pro
                if model == pro_model:

                    # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
                    pred_probs_102 = torch.softmax(vitbase_model_102(image), dim=1)
                    pred_probs_101 = torch.softmax(vitbase_model_101(image), dim=1)

                    # Calculate entropy
                    entropy = -torch.sum(pred_probs_101 * torch.log(pred_probs_101), dim=1).item()

                    # Create a prediction label and prediction probability dictionary for each prediction class
                    pred_classes_and_probs_102 = {class_names_102[i]: float(pred_probs_102[0][i]) for i in range(num_classes_102)}
                    pred_classes_and_probs_101 = {class_names_101[i]: float(pred_probs_101[0][i]) for i in range(num_classes_101)}
                    pred_classes_and_probs_101["unknown"] = 0.0

                    # Get the top predicted class
                    top_class_102 = max(pred_classes_and_probs_102, key=pred_classes_and_probs_102.get)
                    sec_class_102 = sorted(pred_classes_and_probs_102.items(), key=lambda x: x[1], reverse=True)[1][0]
                    top_class_101 = max(pred_classes_and_probs_101, key=pred_classes_and_probs_101.get)

                    # If the image is likely to be an unknown category
                    if pred_probs_101[0][class_names_101.index(top_class_101)] <= MULTICLASS_CLASSIF_THR and entropy > ENTROPY_THR:

                        # Create prediction label and prediction probability for class unknown and rescale the rest of predictions
                        pred_classes_and_probs_101["unknown"] = pred_probs_101.max() * 1.25
                        prob_sum = sum(pred_classes_and_probs_101.values())
                        pred_classes_and_probs = {key: value / prob_sum for key, value in pred_classes_and_probs_101.items()}

                        # Get the top predicted class
                        top_class = "unknown"
                    
                    elif ((top_class_101 == sec_class_102) and (top_class_102 == "unknown")) or (top_class_101 == top_class_102):
                            
                        # Get the probability vector
                        pred_classes_and_probs = pred_classes_and_probs_101

                        # Get the top predicted class
                        top_class = top_class_101

                    else:

                        # Get the probability vector
                        pred_classes_and_probs = pred_classes_and_probs_102

                        # Get the top predicted class
                        top_class = top_class_102

                # Otherwise
                else:

                    # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
                    pred_probs = torch.softmax(vitbase_model_101(image), dim=1) # 101 classes

                    # Calculate entropy
                    entropy = -torch.sum(pred_probs * torch.log(pred_probs), dim=1).item()

                    # Create a prediction label and prediction probability dictionary for each prediction class
                    pred_classes_and_probs = {class_names_101[i]: float(pred_probs[0][i]) for i in range(num_classes_101)}
                    pred_classes_and_probs["unknown"] = 0.0

                    # Get the top predicted class
                    top_class = max(pred_classes_and_probs, key=pred_classes_and_probs.get)

                    # If the image is likely to be an unknown category
                    if pred_probs[0][class_names_101.index(top_class)] <= MULTICLASS_CLASSIF_THR and entropy > ENTROPY_THR:

                        # Create prediction label and prediction probability for class unknown and rescale the rest of predictions
                        pred_classes_and_probs["unknown"] = pred_probs.max() * 1.25
                        prob_sum = sum(pred_classes_and_probs.values())
                        pred_classes_and_probs = {key: value / prob_sum for key, value in pred_classes_and_probs.items()}

                        # Get the top predicted class
                        top_class = "unknown"
                        
            # Otherwise
            else:

                # Set all probabilites to zero except class unknown
                pred_classes_and_probs = {class_names_101[i]: 0.0 for i in range(num_classes_101)}
                pred_classes_and_probs["unknown"] = 1.0
            
                # Get the top predicted class
                top_class = "unknown"

        
        # Get the description of the top predicted class
        top_class_description = food_descriptions.get(top_class, "Description not available.")

        # Calculate the prediction time
        pred_time = f"{round(timer() - start_time, 1)} s."
        
        # Return the prediction dictionary and prediction time 
        return pred_classes_and_probs, pred_time, top_class_description, ""
    
    except Exception as e:
        print(f"[ERROR] {e}")
        error_message = f"<p style='color:red;'>Error during prediction: {str(e)}</p>"
        return {}, "N/A", "N/A", error_message

# Function to handle the model selection state
def handle_model_selection(model):
    return model  # Return the current model

# Configure and design the Gradio App

# Create title, description, and examples
title = "Transform-Eats Large<br>🥪🥗🥣🥩🍝🍣🍰"
description = f"""
A cutting-edge Vision Transformer (ViT) model to classify 101 delicious food types. Discover the power of AI in culinary recognition.

### Supported Food Types:
{', '.join(class_names_102[:-1])}.
"""

# Configure the upload image area
upload_input = gr.Image(
    type="pil",
    label="Upload Image",
    sources=['upload'],
    show_label=True,
    mirror_webcam=False
    )

model_radio = gr.Radio(
    choices=[lite_model, pro_model],
    value=pro_model,
    label="Select the AI algorithm:",
    info="ViT Pro is selected by default if none is chosen."
    )

# Define the status message output field to display error messages
status_output = gr.HTML(label="Status:")

# Set allow flagging
flagging_mode = "never" # "manual"

# Configure the sample image area
# food_vision_examples = [["examples/" + example] for example in os.listdir("examples")]

# Author
article = "Created by Sergio Sanz."

# Create sliders for the thresholds
#prob = gr.Slider(minimum=0, maximum=1, step=0.05, value=0.4, label="Probability Threshold")
#entropy = gr.Slider(minimum=0, maximum=4.615, step=0.5, value=2.5, label="Entropy Threshold")

# Create the Gradio demo
demo = gr.Interface(fn=predict,                                                # mapping function from input to outputs
                    inputs=[upload_input, model_radio],                        # inputs
                    outputs=[gr.Label(num_top_classes=3, label="Prediction"), 
                             gr.Textbox(label="Prediction time:"),
                             gr.Textbox(label="Food Description:"),
                             status_output
                             ],                                                # outputs
                    #examples=food_vision_examples,                            # Create examples list from "examples/" directory
                    #cache_examples=True,                                      # Cache the examples
                    title=title,                                               # Title of the app
                    description=description,                                   # Brief description of the app
                    article=article,                                           # Created by...
                    flagging_mode=flagging_mode,                               # Only For debugging
                    flagging_options=["correct", "incorrect"],                 # Only For debugging
                    theme="ocean")                                             # Theme

# Launch the demo!
demo.launch()
