# 🫀 CardioGuard: End-to-End Heart Disease Prediction System
### 👨‍💻 Developed by: **Najeeb Ullah**

A production-ready Machine Learning system and interactive web dashboard for real-time cardiovascular disease risk assessment, built with **Python**, **Scikit-Learn**, **XGBoost**, **Jupyter Notebook**, and **Streamlit**.

---

## 📁 Project Structure

```
heat_disease_prediction/
│
├── data/
│   └── heart.csv                   # UCI Cleveland Heart Disease dataset (303 records, 14 features)
│
├── models/
│   ├── heart_disease_model.pkl      # Serialized champion model + StandardScaler bundle
│   └── model_metrics.json          # Benchmark evaluation metrics for all trained models
│
├── notebook.ipynb                   # End-to-end Jupyter Notebook (EDA, Visualizations, ML Training)
├── train_model.py                   # Standalone modular ML training and serialization script
├── app.py                           # Streamlit Web Application & Interactive Dashboard
├── requirements.txt                 # Project dependencies list
└── README.md                        # Documentation & setup instructions
```

---

## 🛠️ Step-by-Step Environment Setup & Installation

### 1. Clone or Open the Project Directory
```bash
cd heat_disease_prediction
```

### 2. Create and Activate Virtual Environment

#### 🪟 On Windows:
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

#### 🍎 On macOS / Linux:
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

---

### 3. Install Required Dependencies
Install all required libraries with a single command:
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### 1. Run Machine Learning Training Pipeline
To train all 4 models (Logistic Regression, Random Forest, XGBoost, AdaBoost), evaluate performance metrics, and export the `.pkl` artifact:
```bash
python train_model.py
```

### 2. Launch the Streamlit Web Application
To start the interactive web application:
```bash
streamlit run app.py
```
Open your browser and navigate to: **`http://localhost:8501`**

### 3. Open & Run Jupyter Notebook
To explore the step-by-step EDA, interactive charts, and model benchmarking in Jupyter:
```bash
jupyter notebook notebook.ipynb
```
*(Or open directly in VS Code / Cursor / JupyterLab)*

---

## 📊 Machine Learning Model Benchmarking

All models were evaluated on an **80/20 stratified holdout test split** with **5-Fold Cross-Validation**:

| Model | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC | 5-Fold CV Accuracy |
|---|---|---|---|---|---|---|
| **Random Forest** 🏆 | **91.80%** | **87.10%** | **96.43%** | **91.53%** | **0.9502** | **80.55% (± 0.026)** |
| **XGBoost** | 90.16% | 84.38% | 96.43% | 90.00% | 0.9502 | 80.58% (± 0.025) |
| **AdaBoost** | 88.52% | 83.87% | 92.86% | 88.14% | 0.9610 | 83.89% (± 0.020) |
| **Logistic Regression** | 86.89% | 81.25% | 92.86% | 86.67% | 0.9513 | 82.64% (± 0.017) |

> 🏆 **Champion Model**: **Random Forest** selected for optimal composite performance and high **Recall (96.43%)**, which is crucial in clinical diagnostics to minimize false negatives.

---

## 🩺 Dataset Feature Reference

The dataset includes 13 clinical predictors and 1 binary diagnostic target:

| Feature Name | Description | Values / Units |
|---|---|---|
| `age` | Age of the patient | Years (29 - 77) |
| `sex` | Biological Sex | 1 = Male, 0 = Female |
| `cp` | Chest Pain Type | 1 = Typical Angina, 2 = Atypical, 3 = Non-Anginal, 4 = Asymptomatic |
| `trestbps` | Resting Blood Pressure | mm Hg upon hospital admission |
| `chol` | Serum Cholesterol | mg/dl |
| `fbs` | Fasting Blood Sugar > 120 mg/dl | 1 = True, 0 = False |
| `restecg` | Resting Electrocardiographic Results | 0 = Normal, 1 = ST-T wave abnormality, 2 = LV hypertrophy |
| `thalach` | Maximum Heart Rate Achieved | Beats per minute (bpm) |
| `exang` | Exercise Induced Angina | 1 = Yes, 0 = No |
| `oldpeak` | ST Depression Induced by Exercise | Numerical depression relative to rest |
| `slope` | Slope of Peak Exercise ST Segment | 1 = Upsloping, 2 = Flat, 3 = Downsloping |
| `ca` | Major Vessels Colored by Fluoroscopy | 0 to 3 vessels |
| `thal` | Thalassemia Status | 3 = Normal, 6 = Fixed Defect, 7 = Reversible Defect |
| **`target`** | **Diagnosis of Heart Disease** | **0 = No Disease (Healthy), 1 = Heart Disease Detected** |

---

## 🌟 Streamlit Web Application Features

- 🎛️ **Sidebar Controls**: Easy-to-use sliders, dropdowns, and radio buttons with medical tooltips and range guidance.
- ⚡ **Instant Risk Prediction**: Predicts heart disease status (`0` vs `1`) on a single click.
- 📈 **Interactive Risk Gauge**: Visualizes probability percentage with color-coded risk bands.
- ⚠️ **Clinical Risk Factor Audit**: Automatically identifies abnormal biomarkers (elevated BP, high cholesterol, exercise angina, etc.).
- 💡 **Actionable Recommendations**: Tailored clinical and lifestyle next steps based on risk tier.
- 📊 **Multi-Model Analytics Tab**: Interactive model metric comparisons, confusion matrices, and feature importance rankings.
- 🔬 **Exploratory Data Analysis Tab**: Filterable raw dataset view, distributions, and correlation heatmap.
- 📖 **Clinical Reference Glossary**: Detailed medical descriptions for all 13 parameters.

---

## ⚖️ License & Disclaimer
This project is developed for educational and research purposes only. It is not intended for formal clinical diagnosis without medical professional supervision.
