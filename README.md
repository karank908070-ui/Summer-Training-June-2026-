🚗 Automotive Market Intelligence System

This repository contains an **Automotive Market Intelligence System** built using Machine Learning. The application is designed to help users analyze and evaluate vehicle data through dynamic web-based predictions.

## 📊 Core Machine Learning Models
The system implements two fundamental types of predictive modeling:
1. **Linear Regression (Regression Task):** Utilized to estimate and predict the exact selling price of a car based on structural and performance inputs.
2. **Logistic Regression (Classification Task):** Utilized to categorize the car's market value into either **High Value** or **Low Value** brackets, providing real-time class probabilities and confidence metrics.

## 🛠️ Project Architecture & Features
* **Feature Processing:** Includes robust encoding (`encoders.pkl`) and feature scaling (`scaler_linear.pkl`, `scaler_logistic.pkl`) pipelines to ensure high model accuracy.
* **Interactive UI:** Built an intuitive dashboard using Streamlit allowing real-time input selection (Brand, Age, KM Driven, Fuel Type, Transmission, Max Power, etc.).
* **Data Visualization:** Generates dynamic probability distribution bar charts for classification outcomes directly on the user interface.

## 📂 File Directory Structure
* `train.py` - Source script detailing model training and evaluation phases.
* `appppp.py` - Main application script managing the Streamlit web deployment.
* `cardekho_dataset` - Underlying dataset hosting processed automotive records.
* `*.pkl` - Serialized pre-trained machine learning components, scalers, and encoders.
