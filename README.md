# 🔮 ChurnAI - Customer Churn Prediction Web Application

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gourab007-ai/ChurnAI/blob/main/notebooks/churn_model_training.ipynb)
[![Google Colab Original](https://img.shields.io/badge/Google%20Colab-Original%20Notebook-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/146ynq3ThF4Sfn0PZRPEEPYyRHRXu9uGn?usp=sharing)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask-black.svg)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/Deep%20Learning-TensorFlow%20%2F%20Keras-orange.svg)](https://tensorflow.org/)

**ChurnAI** is an AI-powered end-to-end web application for predicting banking customer churn using deep learning (Artificial Neural Networks / Keras & TensorFlow). It provides an interactive web dashboard for real-time risk assessment, data exploration, feature analysis, and model performance metrics.

---

## 🧠 Model Development in Google Colab

The machine learning pipeline and deep learning neural network were developed, trained, and evaluated in **Google Colab**:
- 🔗 **Interactive Colab Notebook:** [Open in Google Colab](https://colab.research.google.com/github/gourab007-ai/ChurnAI/blob/main/notebooks/churn_model_training.ipynb) or [View Live Shared Colab Notebook](https://colab.research.google.com/drive/146ynq3ThF4Sfn0PZRPEEPYyRHRXu9uGn?usp=sharing)
- **Dataset:** 10,000 banking customer records (`Churn_Modelling.csv`)
- **Exploratory Data Analysis (EDA):** Churn distribution analysis, customer demographic segmentation (Age, Credit Score, Tenure, Balance, Estimated Salary), correlation heatmaps, and outlier checks.
- **Data Preprocessing:**
  - Feature encoding (Label Encoding for Gender, One-Hot Encoding for Geography: France, Germany, Spain).
  - Feature scaling with `StandardScaler` to normalize distributions.
- **Deep Learning Architecture:**
  - Sequential Artificial Neural Network (ANN) built with **TensorFlow / Keras**.
  - Dense layers with **ReLU** activations and **Dropout** regularization to prevent overfitting.
  - Final output layer with **Sigmoid** activation producing continuous churn probabilities.
- **Threshold Calibration:** Tuned decision threshold stored in `threshold.pkl` to optimize Recall and Precision for high-risk customer identification.
- **Exported Production Artifacts:**
  - `churn_model.keras` — Trained deep learning model
  - `scaler.pkl` — Fitted StandardScaler for inference
  - `feature_cols.pkl` — Ordered feature columns schema
  - `threshold.pkl` — Calibrated classification threshold

---

## 🚀 Application Features

- **Real-Time Churn Prediction:** Enter customer details (Credit Score, Age, Tenure, Balance, Number of Products, Has Credit Card, Is Active Member, Estimated Salary, Geography, Gender) to predict whether the customer is likely to churn.
- **Deep Learning Inference:** Real-time scoring using the saved Keras model and threshold calibration.
- **Interactive Web Interface:** Built with Flask, custom CSS styling, and responsive Jinja2 templates.
- **Data & Feature Exploration:** Visualizations and insights into customer demographics, distributions, and feature importance.
- **Model Statistics:** Performance metrics, confusion matrix, and activation function explanations.

---

## 🛠️ Tech Stack

- **Model Training:** Google Colab, TensorFlow 2.x, Keras, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
- **Backend Server:** Flask, Python 3.12+
- **Frontend UI:** HTML5, CSS3, Jinja2 Templates

---

## 📂 Project Structure

```
ChurnAI/
├── notebooks/
│   └── churn_model_training.ipynb   # Complete Google Colab training notebook (EDA + Model + Export)
├── model/
│   ├── churn_model.keras            # Trained Keras neural network model
│   ├── feature_cols.pkl             # Feature column names
│   ├── scaler.pkl                   # StandardScaler object
│   └── threshold.pkl                # Optimal classification threshold
├── static/
│   ├── css/style.css                # Custom styles
│   ├── images/                      # Visual assets and plots
│   └── videos/                      # Demo and background videos
├── templates/
│   ├── index.html                   # Home page
│   ├── about.html                   # About project & architecture
│   ├── data.html                    # Dataset overview
│   ├── features.html                # Feature breakdown
│   ├── predict.html                 # Live prediction interface
│   └── statistics.html              # Model performance & confusion matrix
├── app.py                           # Flask application entry point
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore rules
└── README.md                        # Project documentation
```

---

## ⚡ Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/gourab007-ai/ChurnAI.git
cd ChurnAI
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application
```bash
python app.py
```

Open your browser and navigate to `http://127.0.0.1:5000/`.

---

## 📄 License
This project is open-source and available under the MIT License.
