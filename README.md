# WildFireVision: Deep Learning-Based Wildfire Detection and Prediction

WildFireVision is a deep learning project focused on accurate and real-time wildfire detection and prediction using satellite imagery and weather data. The project leverages state-of-the-art convolutional neural network architectures to classify wildfire presence and provides a user-friendly web interface for rapid risk assessment.

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Project Goals](#project-goals)
- [Methodology](#methodology)
- [Data Acquisition](#data-acquisition)
- [Execution Hierarchy](#execution-hierarchy)
- [Model Selection](#model-selection)
- [Web Application Features](#web-application-features)
- [Project Outcomes](#project-outcomes)
- [Future Work](#future-work)
- [Authors](#authors)
- [Acknowledgments](#acknowledgments)
- [References](#references)

## Overview

Wildfires pose a significant and growing threat to ecosystems, communities, and infrastructure worldwide. Reliable prediction and early detection systems are crucial for effective intervention and mitigation. WildFireVision aims to address this challenge by combining deep learning with multi-source data for robust wildfire prediction[1].

## Problem Statement

The increasing frequency and intensity of wildfires necessitate advanced tools for early warning and risk assessment. This project seeks to build a deep learning model capable of detecting and predicting wildfires from satellite images and weather data, supporting proactive wildfire management.

## Project Goals

- Develop a deep learning model for accurate wildfire detection and prediction.
- Integrate satellite imagery and weather data for improved model performance.
- Deploy the model in a real-time web application for accessible early warning and analysis.

## Methodology

- **Data Acquisition:** Collect and preprocess satellite images and weather data from reliable sources (e.g., Kaggle datasets).
- **Model Design:** Implement and train deep learning models (ResNet50, VGG16, and custom CNNs) with hyperparameter tuning.
- **Model Evaluation:** Assess model performance using accuracy, loss, and confusion matrix metrics.
- **Web Application:** Build an interactive Streamlit app for real-time image-based wildfire prediction and visualization.

## Data Acquisition

Datasets are obtained using Kaggle APIs, including:
- [The Wildfire Dataset](https://www.kaggle.com/datasets/elmadafri/the-wildfire-dataset)
- [FlameVision Dataset](https://www.kaggle.com/datasets/anamibnjafar0/flamevision)

Data preprocessing scripts are provided for organizing and merging datasets into a unified structure for training and evaluation[1].

## Execution Hierarchy

- `model_pred_resnet50.py`: Train and evaluate a ResNet50-based wildfire image classifier.
- `eda.py`: Perform exploratory data analysis and data cleaning.
- `model_pred_vgg16.py`: Train and evaluate a VGG16-based classifier (best-performing model).
- `streamlit_appv3.py`: Launch the web application for real-time wildfire prediction from images.

## Model Selection

- **Basic CNN:** Provided reasonable baseline performance.
- **ResNet50:** High accuracy but prone to overfitting.
- **VGG16:** Achieved the best results with ~90% accuracy and robust generalization.

## Web Application Features

- Upload images for instant wildfire prediction.
- Visualize fire locations and severity.
- Access historical wildfire data and trends.
- Intuitive, user-friendly interface for seamless interaction with the model.

## Project Outcomes

- Developed a high-accuracy deep learning model for wildfire detection.
- Deployed the model in a real-time web application for practical use.
- Demonstrated the effectiveness of deep learning for proactive wildfire management.

## Future Work

- Incorporate additional data sources (e.g., social media, sensor networks).
- Explore ensemble and explainable AI techniques.
- Enhance model interpretability and trust.

## Authors

- Mahikshit Kurapati
- Pooja Chandrashekara
- Mohammad Kanu

## Acknowledgments

- Special thanks to Amir Jafari, Department of Data Science, for guidance.
- Recognition to Kaggle and dataset contributors.

## References

- [IJNRD Paper](https://www.ijnrd.org/papers/IJNRD2305193.pdf)
- [Streamlit Prophet](https://github.com/artefactory/streamlit_prophet)
- [Journal of Big Data Article](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0)
