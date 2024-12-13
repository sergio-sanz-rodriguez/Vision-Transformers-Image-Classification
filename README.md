<div align="center">
  <img src="images/sample_food_images.png" alt="Into Picture" width="1000"/>
</div>

# Implementing Vision Transformers (ViT) for Multi-class Image Classification

## 1. Author

[Sergio Sanz](https://www.linkedin.com/in/sergio-sanz-rodriguez/)

## 1. Overview

This project focuses on the implementation, testing, and evaluation of Vision Transformer (ViT) models using PyTorch. The architecture is based on the groundbreaking paper titled ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929) (see above Figure), which introduced the application of transformers—originally developed for Natural Language Processing (NLP)—to computer vision.

The primary objective is to assess the accuracy and performance of ViT models using the [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) dataset, which consists of 101 food categories. Additionally, a web application showcasing the selected model has been developed to demonstrate its practical use in real-world scenarios.

## 2. Description of the Architecture

A Vision Transformer (ViT) is a state-of-the-art neural network that utilizes the attention mechanism as its primary learning layer. It divides an image into square patches and establishes relationships between them by identifying the most relevant regions based on contextual information. The multi-head attention mechanism processes these patches and generates a sequence of vectors, each representing a patch along with its contextual features.

These vectors are then passed through a series of non-linear multilayer perceptrons (MLPs), which further extract complex relationships between the patches, enabling the model to understand and analyze the image at a high level. 

One of the outputs of the transformer encoder, typically the representation of the classification token (a special token added to the input sequence), is passed to a simple neural network (often a single-layer classifier) that determines the class to which the input image belongs.

<div align="center">
  <img src="images/vit-paper-figure-1-architecture-overview.png" alt="Into Picture" width="1000"/>
</div>

Within the scope of this project, two ViT model architecutes have been implemented and evaluated: **ViT-Base** and **ViT-Large**.

## 3. Description of the Notebooks

* [Custom_Data_Creation.ipynb](https://github.com/sergio-sanz-rodriguez/Vision-Transformers-Image-Classification/blob/main/notebooks/Custom_Data_Creation.ipynb): This notebook downloads and creates the image dataset, splitting the date into train and test subsets.
* [EfficientNetB2_Modelin.ipynb](https://github.com/sergio-sanz-rodriguez/Vision-Transformers-Image-Classification/blob/main/notebooks/EfficientNetB2_Modeling.ipynb): In this notebook, an EfficientNetB2 Convolutional Neural Network (CNN) is trained for different combinations of parameters, such as batch size, hidden units, and number of epochs.
* [Vision_Transformer_Modeling.ipynb](https://github.com/sergio-sanz-rodriguez/Vision-Transformers-Image-Classification/blob/main/notebooks/Vision_Transformer_Modeling.ipynb): This notebook outlines the creation, compilation, and training of multiple ViT-Base and ViT-Large networks, by applying both transfer learning and regular learning of the whole backbones. Several training configurations have been tested in order to find the optimal tunning for these architectures.
* [Model_Evaluation.ipynb](https://github.com/sergio-sanz-rodriguez/Vision-Transformers-Image-Classification/blob/main/notebooks/Model_Evaluation.ipynb): This notebook mainly focuses on evaluating the best performing ViT model obtained from the Vision_Transformer_Modeling.ipynb notebook. The evaluation metrics used include: accuracy, false positive rate at 95% recall, prediction time on the CPU and GPU, model size, and number of parameters.



## 4. The ViT Model In Numbers

