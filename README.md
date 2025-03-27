# 📊 Machine Learning-Based CPU Time Prediction for Jobs

## 🚀 Overview
This project implements various **Machine Learning (ML) models** to predict **CPU time usage** for different jobs based on historical and process attributes.  
The models used include:  
✅ **Random Forest**  
✅ **K-Nearest Neighbors (KNN)**  
✅ **Support Vector Machines (SVM)**  
✅ **Multi-Layer Perceptron (MLP)**  
✅ **REPTree (Decision Tree Regressor)**  

The study also incorporates **feature selection techniques** to identify the most significant attributes for improved prediction accuracy.

---

## 📂 Project Structure  

📁 CPU-Time-Prediction │── 📄 model_comparison.py # Main script for training & evaluating ML models │── 📄 feature_selection.py # Feature selection script using correlation analysis │── 📄 requirements.txt # Required dependencies for the project │── 📄 README.md # Project documentation │── 📊 results/ # Contains generated result files & graphs │── 📄 Jobs_export.csv # Dataset used for model training │── 📄 model_comparison_results.csv # Comparison results of different ML models │── 📊 figures/ # Plots and visualizations



---

## 📜 Methodology  

### 🔹 **1. Data Preprocessing**
- Loaded dataset (`Jobs_export.csv`), removed non-numeric columns.  
- Scaled numerical features using **StandardScaler** for models requiring normalization.  

### 🔹 **2. Feature Selection**
- Applied **correlation analysis** to select the most relevant features.  
- Feature selection helped in reducing noise and improving model performance.  

### 🔹 **3. Model Training & Evaluation**
- Trained multiple models (**RandomForest, KNN, SVM, MLP, and REPTree**).  
- Evaluated using **Mean Absolute Error (MAE) & R² Score**.  

### 🔹 **4. Results Analysis**
- Plotted performance metrics for model comparison.  

---

## 📊 Results Summary  

| Model         | MAE      | R² Score |
|--------------|---------|----------|
| RandomForest | 932.89  | 0.9841   |
| KNN          | 669.81  | 0.9814   |
| SVM          | 1250.54 | 0.9567   |
| MLP          | 2285.44 | 0.9515   |
| REPtree      | 994.63  | 0.9819   |

---

## 📈 Visualizations  
✅ **Feature Importance Graph** (from Feature Selection).  
✅ **Comparison of Model Performance (MAE & R² Score).**  

---

## 🛠 Installation  
### 1️⃣ Clone the repository:  
```sh
git clone https://github.com/your-username/CPU-Time-Prediction.git
cd CPU-Time-Prediction
pip install -r requirements.txt
python model_comparison.py
