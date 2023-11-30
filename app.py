from flask import Flask, render_template, request
import pickle
import joblib
import numpy as np

app = Flask(__name__)

# Load the pickled model
with open('insurance_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the scaler
scaler = joblib.load('scale_model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = float(request.form['age'])
        sex = int(request.form['sex'])
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = int(request.form['smoker'])
        region = int(request.form['region'])

        # Scale the input data
        input_data = np.array([[age, sex, bmi, children, smoker, region]])
        scaled_data = scaler.transform(input_data)

        # Make a prediction using the loaded model
        prediction = model.predict(scaled_data)[0]

        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
