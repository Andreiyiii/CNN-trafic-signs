# Traffic Sign Classifier

This Python project trains a **Convolutional Neural Network (CNN)** to classify traffic sign images into **43 categories** using TensorFlow and OpenCV.

## Features

- Loads and preprocesses image data using OpenCV,resizing images to all be equal size
- Builds and trains a CNN with multiple convolutional layers and max pooling.  
- Supports saving the trained model to a file.  
- Evaluates model accuracy on a test set.

## Model Experiments
- When experimenting with only one single convolution and pooling layer the accuracy was weak ,around 0.054 values with a loss over 3.5
- Seeing this i have added 3 convolution layers with the same settings and the accuracy jumped to around 0.9858 with a loss of 0.0545 saving the model as 3conv1pool.h5
- Afraid of overfitting i have tried using only 2 convolution layers but the accuracy was lower at 0.9753 with a loss of 0.1096
- Since images have a really small dimension(30x30) i have tried no pooling but the results still have lower accuracy and bigger loss compared to 3conv1poll.h5


## Usage
```bash
# Train and evaluate without saving
python traffic.py data_directory

# Train, evaluate, and save the model
python traffic.py data_directory model.h5
