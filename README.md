# [Colab Notebook for Advanced Clothing Classification](https://colab.research.google.com/drive/1zx7YBvSMLuN-_NjvJYi6y0Qp-Am8oqB_?usp=sharing) - README

## Project Overview

This project involves the implementation of various Convolutional Neural Network (CNN) architectures, including the traditional LeNet-5 model and modern transfer learning models like VGG-16 and ResNet-50, for classifying images in the Fashion-MNIST dataset. The project aims to compare the performance of a custom-trained CNN against pre-trained models to evaluate which approach is better suited for the given dataset.

## Dataset Description

The **Fashion-MNIST** dataset consists of 70,000 grayscale images, each of size 28x28 pixels, divided into 10 categories of clothing items:
- **0**: T-shirt/top
- **1**: Trouser
- **2**: Pullover
- **3**: Dress
- **4**: Coat
- **5**: Sandal
- **6**: Shirt
- **7**: Sneaker
- **8**: Bag
- **9**: Ankle boot

Out of these, 60,000 images are used for training and 10,000 for testing.

## Project Structure
- **Data Preparation**: The dataset is first analyzed and preprocessed. This includes checking for missing values, handling duplicates, and visualizing data patterns using correlation matrices and heatmaps.
  
- **Model Development**: Multiple CNN architectures are developed:
  - **LeNet-5**: A simple, custom-built CNN model used for initial experimentation.
  - **VGG-16** and **ResNet-50**: Modern transfer learning models used to evaluate if pre-trained models perform better on this specific task.
  
- **Model Evaluation**: The performance of the models is evaluated using metrics like accuracy and loss. Additionally, 5-fold cross-validation is conducted to get a robust estimate of the model's performance.

## Key Components
### 1. Data Preparation
1. **Loading and Cleaning**:
   - Check for missing values and duplicates.
   - Drop duplicate records to ensure data integrity.
2. **Normalization**: Pixel values are normalized to a range of [0, 1].
3. **Data Visualization**: Correlation matrices and heatmaps are used to analyze the relationship between pixel values.

### 2. Model Training
1. **LeNet-5 Implementation**:
   - Built from scratch using TensorFlow/Keras.
   - Hyperparameter tuning was conducted using a Keras Tuner.
   - 5-fold cross-validation was used for evaluation.

2. **Transfer Learning Models**:
   - **VGG-16** and **ResNet-50** architectures were implemented using transfer learning techniques.
   - The images were resized to 32x32 to match the input requirements of these models.

### 3. Model Comparison
- **LeNet-5** achieved a higher validation accuracy than the transfer learning models. This suggests that custom-trained models may outperform pre-trained models on specific datasets.

## Results
The performance of the models can be summarized as follows:

| Model      | Validation Accuracy | Validation Loss |
|------------|---------------------|-----------------|
| LeNet-5    | 0.8947              | 0.2888          |
| VGG-16     | 0.8768              | 0.3973          |
| ResNet-50  | 0.7875              | 0.5851          |

## Observations
- **LeNet-5** outperformed both VGG-16 and ResNet-50, likely due to being specifically trained from scratch for this dataset, whereas the pre-trained models struggled with adapting to the Fashion-MNIST images.
- The architecture and hyperparameters of LeNet-5 were better suited for extracting features from the relatively simple grayscale images.

## Requirements
- **Libraries**: `numpy`, `pandas`, `matplotlib`, `seaborn`, `tensorflow`, `keras`, `kerastuner`
- **Framework**: Google Colab or a similar Python environment with support for Keras and TensorFlow.

## How to Run
1. Clone the repository and ensure all required libraries are installed.
2. Open the `Advanced Clothing Classification.ipynb` file in Google Colab or click the [Colab Notebook Link](https://colab.research.google.com/drive/1zx7YBvSMLuN-_NjvJYi6y0Qp-Am8oqB_?usp=sharing).
3. Follow the instructions in each code cell to run the project from data loading, preprocessing, model training, and evaluation.

## Conclusion
This project demonstrates the importance of choosing the right architecture for a given problem. While transfer learning is powerful, custom-built models like LeNet-5 can outperform them when designed and tuned specifically for the dataset in use.
