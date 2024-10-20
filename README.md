# ECG Arrhythmia Classification Using Transfer Learning
This repository contains the implementation of an ECG arrhythmia classification model using transfer learning with the AlexNet architecture. The dataset used for this project is the MIT-BIH Arrhythmia dataset, and the model is fine-tuned to classify different arrhythmias using augmented images of ECG signals.
# Dataset
The dataset used for this project is provided in the repo itself.
The dataset is split into training, validation, and testing sets with the following ratios:
Training Set: 70%
Validation Set: 15%
Testing Set: 15%
The dataset is organized into subfolders where each folder represents a different arrhythmia class.

# Preprocessing Steps
1. Data Splitting: The dataset is split into training, validation, and testing sets.
2. Data Augmentation: Augmentation is applied to the training data, including random rotations for better generalization.
3. Resizing: All images are resized to 227x227x3 to match the input size required by the AlexNet model.

# Transfer Learning Setup
The AlexNet model is pre-trained on the ImageNet dataset. In this implementation:

The last three layers of AlexNet (fully connected, softmax, and classification) are replaced with custom layers to adapt to the arrhythmia classification task.
The convolutional layers can be optionally frozen to retain pre-trained features, preventing overfitting, especially if the dataset is small.

# Custom Layers:
Fully Connected Layer: Adapted to the number of classes (5 arrhythmia classes in this case).
Dropout Layer: Added to reduce overfitting.
Softmax Layer: For multi-class classification.
Classification Layer: To compute the final output.

# Training
The model is trained using the Stochastic Gradient Descent with Momentum (SGDM) optimizer with the following hyperparameters:

Mini-Batch Size: 32
Max Epochs: 32
Initial Learning Rate: 1e-4
Validation Frequency: 10
The training process also includes real-time monitoring of the training and validation accuracy.
![Screenshot 2024-10-18 225105](https://github.com/user-attachments/assets/ba09cc33-37a2-452f-9c77-7eb91ce41d7e)

# Evaluation
After training, the model is evaluated on the test dataset. Metrics such as Precision, Recall (Sensitivity), Specificity, and F1 Score are calculated for each class. The overall accuracy of the model is also computed.

# Confusion Matrix
A confusion matrix is plotted to visually inspect the classification performance across all classes.
![Screenshot 2024-10-07 213005](https://github.com/user-attachments/assets/5fc759c8-7545-4ed8-89a1-ec5dd3067458)
You may not get the same confusion matrix.

# Model Saving
The trained model is saved in .mat format:
using this command: save('C:\mit-bih-arrhythmia-database-1.0.0\git_hub.mat', 'netTransfer') # here you need to change the path inorder to save the trained model wherever you want.

# Requirements
MATLAB with the Deep Learning Toolbox
Data Set {spectrogram images of 5 types of Arrhythmia}.
Pre-trained AlexNet model (you need to install this add on)

I have also added another file through which you can check your saved model.



