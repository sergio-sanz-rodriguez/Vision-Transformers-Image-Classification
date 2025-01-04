
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
    class_names = f.read().splitlines()

# Specify number of classes
num_classes = len(class_names) # 101 + unknown

# Load the food description file
food_descriptions_json = "food_descriptions.json"
with open(food_descriptions_json, 'r') as f:
    food_descriptions = json.load(f)

# Instantiate the model
classification_model_name_path = "effnetb0_classif_epoch13.pth"
effnetb0_model = create_effnetb0(
    model_weights_dir=".",
    model_weights_name=classification_model_name_path,
    num_classes=2
    )

# Load the ViT-Base/16 transformer with input image of 384x384 pixels and 101 + unknown classes
vitbase_model = create_vitbase_model(
    model_weights_dir=".",
    model_weights_name="vitbase16_102_2025-01-04_best.pth",
    img_size=384,
    num_classes=num_classes,
    compile=True
)

# Specify manual transforms for model_2
transforms = v2.Compose([    
    v2.Resize(384), #v2.Resize((384, 384)),
    v2.CenterCrop((384, 384)),    
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]) 
])


# Put models into evaluation mode and turn on inference mode
effnetb0_model.eval()
vitbase_model.eval()

# Predict function
def predict(image) -> Tuple[Dict, str, str]:

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
            if effnetb0_model(image)[:,1].cpu() >= 0.9981166124343872:

                # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
                pred_probs = torch.softmax(vitbase_model(image), dim=1)

                # Calculate entropy
                entropy = -torch.sum(pred_probs * torch.log(pred_probs), dim=1).item()

                # Create a prediction label and prediction probability dictionary for each prediction class
                pred_classes_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(num_classes)}

                # Get the top predicted class
                top_class = max(pred_classes_and_probs, key=pred_classes_and_probs.get)

                # If the image is likely to be an unknown category
                if pred_probs[0][class_names.index(top_class)] <= 0.5 and entropy > 2.6:

                    # Create prediction label and prediction probability for class unknown and rescale the rest of predictions
                    pred_classes_and_probs["unknown"] = pred_probs.max() * 1.25
                    prob_sum = sum(pred_classes_and_probs.values())
                    pred_classes_and_probs = {key: value / prob_sum for key, value in pred_classes_and_probs.items()}

                    # Get the top predicted class
                    top_class = "unknown"

            # Otherwise
            else:

                # Set all probabilites to zero except class unknown
                pred_classes_and_probs = {class_names[i]: 0.0 for i in range(num_classes)}
                pred_classes_and_probs["unknown"] = 1.0
            
                # Get the top predicted class
                top_class = "unknown"
        
        # Get the description of the top predicted class
        top_class_description = food_descriptions.get(top_class, "Description not available.")

        # Calculate the prediction time
        pred_time = f"{round(timer() - start_time, 1)} s."
        
        # Return the prediction dictionary and prediction time 
        return pred_classes_and_probs, pred_time, top_class_description
    
    except Exception as e:
        print(f"[ERROR] {e}")
        return {}, "Error during prediction.", "N/A"

# Configure and design the Gradio App

# Create title, description, and examples
title = "Transform-Eats Large<br>🥪🥗🥣🥩🍝🍣🍰"
description = f"""
A cutting-edge Vision Transformer (ViT) model to classify 101 delicious food types. Discover the power of AI in culinary recognition.

### Supported Food Types
{', '.join(class_names[:-1])}.
"""

# Configure the upload image area
upload_input = gr.Image(type="pil", label="Upload Image", sources=['upload'], show_label=True, mirror_webcam=False)

# Configure the sample image area
food_vision_examples = [["examples/" + example] for example in os.listdir("examples")]

# Author
article = "Created by Sergio Sanz."

# Create sliders for the thresholds
#prob = gr.Slider(minimum=0, maximum=1, step=0.05, value=0.4, label="Probability Threshold")
#entropy = gr.Slider(minimum=0, maximum=4.615, step=0.5, value=2.5, label="Entropy Threshold")

# Create the Gradio demo
demo = gr.Interface(fn=predict,                                                # mapping function from input to outputs
                    inputs=upload_input,                                       # inputs #[upload_input, model_dropdown]
                    outputs=[gr.Label(num_top_classes=3, label="Prediction"), 
                             gr.Textbox(label="Prediction time:"),
                             gr.Textbox(label="Food Description:")],           # outputs
                    examples=food_vision_examples,                             # Create examples list from "examples/" directory
                    cache_examples=True,                                       # Cache the examples
                    title=title,                                               # Title of the app
                    description=description,                                   # Brief description of the app
                    article=article,                                           # Created by...
                    theme="ocean")                                             # Theme

# Launch the demo!
demo.launch()
