
# 1. Imports and class names setup
import os
import torch
import json
import gradio as gr
from model import create_vitbase_model
from timeit import default_timer as timer
from typing import Tuple, Dict
from torchvision.transforms import v2


# 2. Specify class names
food_vision_class_names_path = "class_names.txt"
with open(food_vision_class_names_path, "r") as f:
    class_names = f.read().splitlines()


# 3. Load the food description file
food_descriptions_json = "food_descriptions.json"
with open(food_descriptions_json, 'r') as f:
    food_descriptions = json.load(f)

# 4. Load the ViT-Base transformer
food_vision_model_name_path = "vitbase16_5.pth"
IMG_SIZE = 224
num_classes = len(class_names)
vitbase_model = create_vitbase_model(
    model_weights_dir=".",
    model_weights_name=food_vision_model_name_path,
    img_size=IMG_SIZE,
    num_classes=num_classes
)

# 5. Specify manual transforms
transforms = v2.Compose([    
    v2.Resize((242, 242)),
    v2.CenterCrop((IMG_SIZE, IMG_SIZE)),    
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]) 
])

# 6. Predict function
def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken.
    """
    # Start the timer
    start_time = timer()
    
    # Transform the target image and add a batch dimension
    img = transforms(img).unsqueeze(0)
    
    # Put model into evaluation mode and turn on inference mode
    vitbase_model.eval()
    with torch.inference_mode():

        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax(vitbase_model(img), dim=1)
    
    # Create a prediction label and prediction probability dictionary for each prediction class
    # (this is the required format for Gradio's output parameter)
    pred_classes_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

    # Get the top predicted class and its description
    top_class = max(pred_classes_and_probs, key=pred_classes_and_probs.get)
    #top_class_description = food_descriptions[top_class]
    top_class_description = food_descriptions.get(top_class, "Description not available.")
    
    # Calculate the prediction time
    pred_time = round(timer() - start_time, 2)
    
    # Return the prediction dictionary and prediction time 
    return pred_classes_and_probs, pred_time, top_class_description

# 7. Configure and design the Gradio App
# Create title, description, and examples
title = "Transform-Eats Large 🥪🥗🥩"
description = f"""
A cutting-edge Vision Transformer (ViT) model to classify 101 delicious food types. Discover the power of AI in culinary recognition.

### Supported Food Types
{', '.join(class_names)}.
"""

food_vision_examples = [["examples/" + example] for example in os.listdir("examples")]

article = "Created by Sergio Sanz."

# Create the Gradio demo
demo = gr.Interface(fn=predict,                                                # mapping function from input to outputs
                    inputs=gr.Image(type="pil"),                               # input
                    outputs=[gr.Label(num_top_classes=3, label="Predictions"), # outputs
                             gr.Number(label="Prediction time (s)"),
                             gr.Textbox(label="Food Description")],
                    examples=food_vision_examples,                             # Create examples list from "examples/" directory
                    title=title,                                               # Title of the app
                    description=description,                                   # Brief description of the app
                    article=article,                                           # Created by...
                    theme="ocean")                                             # Theme

# Launch the demo!
demo.launch()
