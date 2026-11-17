from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

app = Flask(__name__)

# ── Load model, scaler, threshold ─────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
model     = tf.keras.models.load_model(os.path.join(BASE, 'model', 'churn_model.keras'))

with open(os.path.join(BASE, 'model', 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

with open(os.path.join(BASE, 'model', 'threshold.pkl'), 'rb') as f:
    threshold = pickle.load(f)

print(f"Model loaded | Threshold: {threshold:.2f}")

# ── Routes ─────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/data')
def data():
    return render_template('data.html')

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/statistics')
def statistics():
    return render_template('statistics.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')

    try:
        form = request.form

        # ── Raw inputs (matching original form field names) ────
        credit_score  = float(form['CreditScore'])
        gender        = float(form['Gender'])          # Male=1, Female=0
        age           = float(form['Age'])
        tenure        = float(form['Tenure'])
        balance       = float(form['Balance'])
        has_cr_card   = float(form['HasCrCard'])       # yes=1, no=0
        is_active     = float(form['IsActiveMember'])  # yes=1, no=0
        salary        = float(form['EstimatedSalary'])

        # Products — one-hot style inputs
        pro1 = float(form['NumOfProducts_1'])
        pro2 = float(form['NumOfProducts_2'])
        pro3 = float(form['NumOfProducts_3'])
        pro4 = float(form['NumOfProducts_4'])

        # Derive NumOfProducts from which one is selected
        if   pro1 == 1: num_products = 1.0
        elif pro2 == 1: num_products = 2.0
        elif pro3 == 1: num_products = 3.0
        elif pro4 == 1: num_products = 4.0
        else:           num_products = 1.0

        # Geography — one-hot style inputs
        geo_france  = float(form['Geography_0'])   # France=1
        geo_germany = float(form['Geography_1'])   # Germany=1
        geo_spain   = float(form['Geography_3'])   # Spain=1

        # ── Engineered features (must match Colab training) ────
        balance_salary_ratio  = balance / (salary + 1)
        age_group             = int(pd.cut([age], bins=[0,30,40,50,60,100], labels=[0,1,2,3,4])[0])
        tenure_age_ratio      = tenure / (age + 1)
        products_per_tenure   = num_products / (tenure + 1)
        median_balance        = 97198.54
        is_high_value         = 1.0 if (balance > median_balance and num_products >= 2) else 0.0
        zero_balance          = 1.0 if balance == 0 else 0.0

        raw = np.array([[
            credit_score, gender, age, tenure, balance,
            num_products, has_cr_card, is_active, salary,
            geo_france, geo_germany, geo_spain,
            balance_salary_ratio, age_group,
            tenure_age_ratio, products_per_tenure,
            is_high_value, zero_balance
        ]])

        scaled = scaler.transform(raw)
        prob   = float(model.predict(scaled, verbose=0)[0][0]) * 100
        prob   = round(prob, 2)

        # Output text exactly matching original repo style
        predict_text = f"The probability of this type of Customer leaving the bank is {prob}%"
        sug_text     = "Explore more on"

        return render_template('predict.html',
            predict_text=predict_text,
            sug_text=sug_text
        )

    except Exception as e:
        predict_text = f"Error: {str(e)}"
        return render_template('predict.html', predict_text=predict_text, sug_text="")


if __name__ == '__main__':
    app.run(debug=True)