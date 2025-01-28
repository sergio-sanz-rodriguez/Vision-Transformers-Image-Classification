
# Imports and class names setup
import os
import torch
import json
import string
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

# Instantiate the classfication model for recognizing between food and non-food images
classification_model_name_path_1 = "effnetb0_2025-01-05_epoch13.pth"
effnetb0_model_1 = create_effnetb0(
    model_weights_dir=".",
    model_weights_name=classification_model_name_path_1,
    num_classes=2,
    compile=True,
    dropout=0.2
    )

# Instantiate the model for recognizing between known and unknown food images
classification_model_name_path_2 = "effnetb0_2_2025-01-12_epoch13.pth"
effnetb0_model_2 = create_effnetb0(
    model_weights_dir=".",
    model_weights_name=classification_model_name_path_2,
    num_classes=2,
    compile=True,
    dropout=0.0
    )

# Load the ViT-Base/16 transformer with input image of 384x384 pixels and 101 + unknown classes
vitbase_model_101 = create_vitbase_model(
    model_weights_dir=".",
    model_weights_name="vitbase16_101_2025-01-27_epoch17.pth",
    image_size=384,
    num_classes=num_classes_101,
    compile=True
)

vitbase_model_102 = create_vitbase_model(
    model_weights_dir=".",
    model_weights_name="vitbase16_102_2025-01-27_epoch19.pth",
    image_size=384,
    num_classes=num_classes_102,
    compile=True
)

# Specify manual transforms for ViTs
transforms_eff = v2.Compose([    
    v2.Resize((256, 256)),
    v2.CenterCrop((224, 224)),    
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]) 
])

# Specify manual transforms for ViTs
transforms_vit = v2.Compose([    
    v2.Resize((384)),
    v2.CenterCrop((384, 384)),    
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]) 
])


# Put models into evaluation mode and turn on inference mode
effnetb0_model_1.eval()
effnetb0_model_2.eval()
vitbase_model_101.eval()
vitbase_model_102.eval()

# Set thresdholds
BINARY_CLASSIF_THR_1 = 0.8310611844062805
BINARY_CLASSIF_THR_2 = 0.06102316826581955 # 41% FPR
#BINARY_CLASSIF_THR_2 = 0.0728086531162262  # 23% FPR
MULTICLASS_CLASSIF_THR = 0.5
ENTROPY_THR = 2.7

# Set model names
lite_model = "⚡ ViT Lite ⚡ faster, less accurate prediction"
pro_model =  "💎 ViT Pro 💎 slower, more accurate prediction"

# Computes the entropy
def entropy(pred_probs):

    """
    Computes the entropy of the predicted probabilities.

    Args:
        pred_probs (torch.Tensor): A tensor containing the predicted probabilities.

    Returns:
        float: The entropy value.
    """
    #pred_probs = pred_probs[1:-1]
    return -torch.sum(pred_probs * torch.log(pred_probs)).item()

# Computes the model prediction outputs as probabilities
def predict(image, model):

    """
    Computes the predicted class probabilities for a given image using the provided model.

    Args:
        image (torch.Tensor): Input tensor representing the image or batch of images.
                              The tensor should be preprocessed as required by the model.
        model (torch.nn.Module): The trained model used to make predictions.

    Returns:
        torch.Tensor: A tensor containing the probabilities for each class. 
                      The output is normalized using the softmax function.
    """
    
    return torch.softmax(model(image), dim=1)

# Predict method
def classify_food(image, model=pro_model) -> Tuple[Dict, str, str]:

    """
    Transforms and performs a prediction on the image and returns prediction details.

    Args:
        image (torch.Tensor): Input tensor representing the image.
                              It should be preprocessed as required by the model.
        model (torch.nn.Module, optional): The trained model used for predictions.
                                           Defaults to pro_model.

    Returns:
        Tuple[Dict, str, str]: A tuple containing:
            - Dictionary of predicted class probabilities.
            - Time taken for prediction as a string.
            - Description of the top predicted class.
    """

    try:

        # Start the timer
        start_time = timer()

        # Transform the target image and add a batch dimension
        image_eff = transforms_eff(image).unsqueeze(0)

        # Check out model parameter
        if model == None:
            model = pro_model
        
        # Make prediction...
        with torch.inference_mode():
            
            # If the picture is food
            if predict(image_eff, effnetb0_model_1)[:,1] >= BINARY_CLASSIF_THR_1:

                # 💎 ViT Pro 💎
                if model == pro_model:

                    # If the image is likely to be an known category
                    if  predict(image_eff, effnetb0_model_2)[:,1] >= BINARY_CLASSIF_THR_2:

                        # Preproces the image for the ViTs
                        image_vit = transforms_vit(image).unsqueeze(0)

                        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
                        pred_probs_102 = predict(image_vit, vitbase_model_102)
                        pred_probs_101 = predict(image_vit, vitbase_model_101)
                        
                        # Calculate entropy
                        entropy_102 = entropy(pred_probs_102)
                        entropy_101 = entropy(pred_probs_101)
                        
                        # Create a prediction label and prediction probability dictionary for each prediction class
                        pred_classes_and_probs_102 = {class_names_102[i]: float(pred_probs_102[0][i]) for i in range(num_classes_102)}
                        pred_classes_and_probs_101 = {class_names_101[i]: float(pred_probs_101[0][i]) for i in range(num_classes_101)}
                        pred_classes_and_probs_101["unknown"] = 0.0
                        
                        # Get the top predicted class
                        top_class_102 = max(pred_classes_and_probs_102, key=pred_classes_and_probs_102.get)
                        sec_class_102 = sorted(pred_classes_and_probs_102.items(), key=lambda x: x[1], reverse=True)[1][0]
                        top_class_101 = max(pred_classes_and_probs_101, key=pred_classes_and_probs_101.get)

                        # Check out entropy
                        condition_102 = pred_probs_102[0][class_names_102.index(top_class_102)] <= MULTICLASS_CLASSIF_THR and entropy_102 > ENTROPY_THR
                        condition_101 = pred_probs_101[0][class_names_101.index(top_class_101)] <= MULTICLASS_CLASSIF_THR and entropy_101 > ENTROPY_THR
                        if condition_101 and condition_102:

                            # Create prediction label and prediction probability for class unknown and rescale the rest of predictions
                            pred_classes_and_probs_101["unknown"] = pred_probs_101.max() * 1.25
                            prob_sum = sum(pred_classes_and_probs_101.values())
                            pred_classes_and_probs = {key: value / prob_sum for key, value in pred_classes_and_probs_101.items()}

                            # Get the top predicted class
                            top_class = "unknown"
                        
                        # Compare the predictions of the two transformer models                    
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

                    # The food is unknown
                    else:

                        # Set all probabilites to zero except class unknown
                        pred_classes_and_probs = {class_names_101[i]: 0.0 for i in range(num_classes_101)}
                        pred_classes_and_probs["unknown"] = 1.0
            
                        # Get the top predicted class
                        top_class = "unknown"

                # ⚡ ViT Lite ⚡
                else:

                    # Preproces the image for the ViTs
                    image_vit = transforms_vit(image).unsqueeze(0)

                    # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
                    pred_probs_101 = predict(image_vit, vitbase_model_101) # 101 classes
                    
                    # Calculate entropy
                    entropy_101 = entropy(pred_probs_101)

                    # Create a prediction label and prediction probability dictionary for each prediction class
                    pred_classes_and_probs = {class_names_101[i]: float(pred_probs_101[0][i]) for i in range(num_classes_101)}
                    pred_classes_and_probs["unknown"] = 0.0

                    # Get the top predicted class
                    top_class = max(pred_classes_and_probs, key=pred_classes_and_probs.get)

                    # If the image is likely to be an unknown category
                    if pred_probs_101[0][class_names_101.index(top_class)] <= MULTICLASS_CLASSIF_THR and entropy_101 > ENTROPY_THR:
                 
                        # Create prediction label and prediction probability for class unknown and rescale the rest of predictions
                        pred_classes_and_probs["unknown"] = pred_probs_101.max() * 1.25
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

#######################################
# Configure and design the Gradio App #
#######################################

# Create title, description, and examples
title = "Transform-Eats Large<br>🥪🥗🥣🥩🍝🍣🍰"
description = f"""
A cutting-edge Vision Transformer (ViT) model to classify 101 delicious food types. Discover the power of AI in culinary recognition.
"""

# Group food items alphabetically
grouped_foods = {letter: [] for letter in string.ascii_uppercase}
for food in class_names_102:
    first_letter = food[0].upper()  # Get the first letter and make it uppercase
    if first_letter in grouped_foods:
        grouped_foods[first_letter].append(food)

# Function to display food items based on button click
def display_foods(letter):
    items = grouped_foods.get(letter, [])
    return f"**{letter}:** {', '.join(items)}" if items else f"No items for {letter}"

# Configure the Gradio interface
with gr.Blocks(theme="ocean") as demo:

    # Title and description (at the top)
    gr.Markdown(f"<h1>{title}</h1>")
    gr.Markdown(f"<p>{description}</p>")

    # Title for supported meals
    supported_meals_title = gr.Markdown("### Supported Meals")

    # Output display area
    output = gr.Markdown()

    # Add the supported meals title and buttons in the layout
    with gr.Column():

        # Keep the title at the top
        supported_meals_title  
        
        with gr.Row():
            buttons = []
            for letter in string.ascii_uppercase:
                button = gr.Button(letter, elem_id=f"button_{letter}", size="sm", min_width=40)
                button.click(display_foods, inputs=[gr.Textbox(value=letter, visible=False)], outputs=output)
                buttons.append(button)

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
    flagging_mode = "never"  # "manual"

    # Author
    article = "Created by Sergio Sanz."

    # Create the Gradio demo
    gr.Interface(
        fn=classify_food,                                                  # mapping function from input to outputs
        inputs=[upload_input, model_radio],                                # inputs
        outputs=[gr.Label(num_top_classes=3, label="Prediction"), 
                 gr.Textbox(label="Prediction time:"),
                 gr.Textbox(label="Food Description:"),
                 status_output],                                           # outputs
                article=article,                                           # Created by...
                flagging_mode=flagging_mode,                               # Only For debugging
                flagging_options=["correct", "incorrect"],                 # Only For debugging
                )  

# Launch the demo!
demo.launch()
