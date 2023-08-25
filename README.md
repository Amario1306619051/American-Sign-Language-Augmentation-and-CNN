# American Sign Language (ASL) Recognition using Convolutional Neural Networks (CNN)


This repository contains code and resources for building a Convolutional Neural Network (CNN) to classify American Sign Language (ASL) images. The module covers various stages including data preprocessing, CNN model creation, training, evaluation, and using the trained model for recognizing ASL letters with high accuracy. It also introduces data augmentation techniques to reduce overfitting and provides practical examples of making predictions using the trained model on new, unseen images.

## Contents

- [Introduction](#introduction)
- [Data Preprocessing](#data-preprocessing)
- [Creating the CNN Model](#creating-the-cnn-model)
- [Data Augmentation](#data-augmentation)
- [Model Predictions](#model-predictions)
- [Usage](#usage)
- [Future Applications](#future-applications)
- [Getting Started](#getting-started)
- [License](#license)

## Introduction

American Sign Language (ASL) is a vital means of communication for the hearing-impaired community. This project aims to build a model that can recognize ASL letters from images, enabling communication through technology. Convolutional Neural Networks (CNNs) are particularly well-suited for image classification tasks, making them a suitable choice for this project.

## Data Preprocessing

Before training the model, the dataset is loaded, preprocessed, and split into training and validation sets. The images are converted to grayscale, normalized, and reshaped to match the input requirements of the CNN model.

## Creating the CNN Model

A CNN model is created using TensorFlow's Keras API. The model consists of multiple layers including convolutional layers, max-pooling layers, dropout layers, batch normalization layers, and dense layers. The architecture is designed to detect important features in ASL images and classify them accurately.

## Data Augmentation

To improve model generalization and reduce overfitting, data augmentation techniques are applied. ImageDataGenerator from Keras is used to perform augmentation such as rotation, zooming, and horizontal shifting on the training data. This increases the dataset's size and variability, making the model more robust.

## Model Predictions

The trained model is loaded and used for making predictions on new, unseen ASL images. Images are preprocessed, resized, normalized, and fed into the model. The model predicts the corresponding ASL letter with high accuracy.

## Usage

To use the trained model for ASL letter recognition on new images:
1. Clone this repository: `https://github.com/Amario1306619051/American-Sign-Language-Augmentation-and-CNN`
2. Navigate to the repository: `cd asl-recognition`
3. Ensure you have the required dependencies installed (TensorFlow, Keras, etc.).
4. Place your ASL images in the `data/asl_images` folder.
5. Run the provided Python script for making predictions: `python predict_asl.py`.

## Future Applications

The trained ASL recognition model can be integrated into various applications:
- Developing educational tools for teaching sign language.
- Creating tools for communication between hearing-impaired individuals and computers.
- Building mobile apps that allow users to translate sign language to text or speech.

## Getting Started

To get started, follow the steps in the [Usage](#usage) section to clone the repository, prepare the environment, and use the trained model for predictions.

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use and modify the code for your own applications.

---

Feel free to contribute to this repository by improving the model, adding new features, or suggesting enhancements. Your contributions can make this project more impactful and useful for the community.