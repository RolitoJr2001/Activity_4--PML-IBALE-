import pickle

import pandas as pd
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

model = pickle.load(open('models/model_act4.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    # feature = request.form.values('spending')
    feature = [float(x) for x in request.form.values()]
    features = [np.array(feature)]
    prediction = model.predict(features)
    print(prediction)

    result = round(prediction[0], 2)

    return render_template('index.html', prediction_output=f'The profit is {result}')


if __name__ == "__main__":
    app.run()