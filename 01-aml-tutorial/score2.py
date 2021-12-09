import json
import os
import joblib

def init():
    global model
    print("This is init")
    model_filename = 'diabetes_logistic_regression.pkl'
    model_path = os.path.join(os.environ['AZUREML_MODEL_DIR'], model_filename)

    model = joblib.load(model_path)


def run(data):
    
    test = json.loads(data)
    result = model.predict(test)
    print(f"received data {test}")
    return test
