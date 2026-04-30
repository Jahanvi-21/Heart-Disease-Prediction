# 🫀 Heart Disease Patient Segmentation Dashboard

An interactive Machine Learning dashboard built with **Streamlit** to segment patients based on cardiovascular risk factors using Unsupervised Learning techniques (**K-Means** vs. **Expectation-Maximization**).

## 🚀 Live Demo
*(Once you deploy on Streamlit Cloud, paste your link here)*
Example: [Live App Link](https://straining-italics-degraded.ngrok-free.dev)

## 📌 Project Overview
This project focuses on identifying distinct patient groups from a heart disease dataset to assist in clinical decision-making. By applying clustering algorithms, we can categorize patients into different health profiles based on physiological indicators.

### Key Features:
- **Comparative Analysis**: Real-time comparison between Distance-based (K-Means) and Distribution-based (EM/GMM) clustering.
- **Dimensionality Reduction**: Implementation of **PCA (Principal Component Analysis)** to visualize 6D data in a 2D interactive space.
- **Performance Metrics**: Evaluation using **Silhouette Scores** and **Model Inertia**.
- **Dynamic UI**: Adjustable cluster counts (k) to see how patient segments evolve.

## 🛠️ Tech Stack
- **Language**: Python 3.x
- **Frontend**: Streamlit
- **ML Libraries**: Scikit-Learn, NumPy, Pandas
- **Visualization**: Matplotlib, PCA

## 📊 Analyzed Features
The models analyze the following cardiovascular indicators:
1. Age
2. Blood Pressure
3. Cholesterol Level
4. BMI (Body Mass Index)
5. Sleep Hours
6. Triglyceride Level

## 🧬 Algorithms Explained
- **K-Means Clustering**: Uses Euclidean distance to create "Hard Clusters." Best for finding distinct, spherical groups.
- **Expectation-Maximization (GMM)**: Uses probability distributions to create "Soft Clusters." Ideal for overlapping patient data where a patient might belong to multiple risk categories with varying probabilities.

## 🏃 How to Run Locally

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/your-username/heart-disease-segmentation.git](https://github.com/your-username/heart-disease-segmentation.git)
   cd heart-disease-segmentation

