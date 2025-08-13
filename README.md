# Flower Image Classifier 🌸

This project involves building and training a deep learning model to classify different species of flowers. Using a pre-trained VGG-16 model, I developed a classifier that can identify 102 different flower categories with high accuracy. The final product is a command-line application that can predict the species of a flower from an image.

This project was completed as part of the Udacity AI Programming with Python Nanodegree.

## Project Overview

The project is broken down into three main parts:

1.  **Data Loading and Preprocessing**: The image dataset is loaded and transformed for optimal training. This includes random scaling, cropping, and flipping for the training set to help the network generalize better.
2.  **Model Training**: A pre-trained VGG-16 network is used as a feature extractor. A new feed-forward classifier is defined, trained, and attached to the VGG-16 model.
3.  **Inference**: The trained classifier is used to predict the class of a flower in an image and display the top *K* most likely classes.

## Dataset

The model was trained on a dataset of 102 flower categories. The dataset is split into training, validation, and testing sets. The images are normalized using the ImageNet dataset's means and standard deviations to match the pre-trained network's expectations. A `cat_to_name.json` file is used to map the category labels to the actual flower names.

## Model Architecture

  * **Pre-trained Network**: The VGG-16 model, pre-trained on ImageNet, is used for feature extraction. The feature parameters are frozen to prevent them from being updated during training.
  * **Custom Classifier**: A new feed-forward network is defined to serve as the classifier. It consists of the following layers:
      * Linear layer (25088 in, 4096 out)
      * ReLU activation
      * Dropout (p=0.5)
      * Linear layer (4096 in, 1024 out)
      * ReLU activation
      * Dropout (p=0.5)
      * Linear layer (1024 in, 102 out)
      * LogSoftmax output layer

## Training and Results

  * **Optimizer**: Adam optimizer with a learning rate of 0.001.
  * **Loss Function**: Negative Log-Likelihood Loss (NLLLoss).
  * **Epochs**: The model was trained for 3 epochs.

The model achieved the following performance:

  * **Validation Accuracy**: 81.2%
  * **Test Accuracy**: 79.1%

## Usage

### Prerequisites

  * Python 3.10+
  * PyTorch
  * NumPy
  * Matplotlib
  * Pillow

### Inference

To make predictions on a new image, you can use the `predict` function, which takes an image path and the trained model as input. It returns the top 5 most likely flower classes and their corresponding probabilities.

```python
# Load the checkpoint
model = load_checkpoint('checkpoint.pth')

# Predict the class from an image file
image_path = 'path/to/your/image.jpg'
probs, classes = predict(image_path, model)

print(probs)
print(classes)
```
