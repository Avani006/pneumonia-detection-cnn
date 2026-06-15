# pneumonia-detection-cnn

Overview

This project focuses on detecting pneumonia from chest X-ray images using a Convolutional Neural Network (CNN). The model learns important visual patterns such as lung opacity and abnormalities directly from images, eliminating the need for manual feature extraction. It classifies X-rays into Normal or Pneumonia, helping in early and automated diagnosis. The goal is to build an accurate and reliable model with high recall, as missing pneumonia cases can be critical in real-world scenarios.

Tech Stack

Python
TensorFlow / Keras
NumPy, OpenCV

Dataset

Chest X-ray dataset (~5,800 images)
Classes: Normal, Pneumonia

Model

CNN architecture
Conv → ReLU → MaxPooling → Dense → Sigmoid

Preprocessing

Image resizing
Normalization (0–1 scaling)
Data augmentation

Training

Loss: Binary Cross-Entropy
Optimizer: Adam

Run Project

git clone <repo-link>
cd pneumonia-detection
pip install -r requirements.txt
python train.py
