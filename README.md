# 🔮 ChurnAI - Customer Churn Prediction Web Application

**ChurnAI** is an AI-powered end-to-end web application for predicting banking customer churn using deep learning (Artificial Neural Networks / Keras & TensorFlow). It provides an interactive web dashboard for real-time risk assessment, data exploration, feature analysis, and model performance metrics.

---

## 🚀 Features

- **Real-Time Churn Prediction:** Enter customer details (Credit Score, Age, Tenure, Balance, Number of Products, Has Credit Card, Is Active Member, Estimated Salary, Geography, Gender) to predict whether the customer is likely to churn.
- **Deep Learning Model:** Powered by a trained Artificial Neural Network (`churn_model.keras`) with calibrated decision threshold and standard scaling.
- **Interactive Web Interface:** Built with Flask, custom CSS styling, and responsive templates.
- **Data & Feature Exploration:** Visualizations and insights into customer demographics, distributions, and feature importance.
- **Model Statistics:** Performance metrics, confusion matrix, and activation function explanations.

---

## 🛠️ Tech Stack

- **Backend:** Flask, Python 3.12+
- **Machine Learning / Deep Learning:** TensorFlow, Keras, Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Frontend:** HTML5, CSS3, Jinja2 templates

---

## 📂 Project Structure

```
ChurnAI/
├── model/
│   ├── churn_model.keras    # Trained Keras neural network model
│   ├── feature_cols.pkl     # Feature column names
│   ├── scaler.pkl           # StandardScaler object
│   └── threshold.pkl        # Optimal classification threshold
├── static/
│   ├── css/style.css        # Custom styles
│   ├── images/              # Visual assets and plots
│   └── videos/              # Demo and background videos
├── templates/
│   ├── index.html           # Home page
│   ├── about.html           # About project & architecture
│   ├── data.html            # Dataset overview
│   ├── features.html        # Feature breakdown
│   ├── predict.html         # Live prediction interface
│   └── statistics.html      # Model performance & confusion matrix
├── app.py                   # Flask application entry point
├── requirements.txt         # Python dependencies
├── .gitignore               # Git ignore rules
└── README.md                # Project documentation
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
