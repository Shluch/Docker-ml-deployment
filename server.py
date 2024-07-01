from fastapi import FastAPI
import joblib
import numpy as np

model = joblib.load('model.joblib')

class_names = np.array(['setosa','versicolor','virginica'])

app = FastAPI

@app.get('/')
def read_root():
    return{'message': 'Iris model API'}

@app.post('/predict')
def predict(data: dict):
    features = np.array(data['features'].reshape(1,-1))
    prediction = model.predict(features)
    class_names = class_names[prediction][0]
    return {'predicted_class': class_names}