import json
import os
import joblib
import numpy as np

# from inference_schema.schema_decorators import input_schema, output_schema
# from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType

def init():
    print("This is init")
    global model

    # The AZUREML_MODEL_DIR environment variable indicates
    # a directory containing the model file you registered.
    model_filename = 'diabetes_logistic_regression.pkl'
    model_path = os.path.join(os.environ['AZUREML_MODEL_DIR'], model_filename)

    model = joblib.load(model_path)



# @input_schema('data', NumpyParameterType(np.array([[0.1, 1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 9.0]])))
# @output_schema(NumpyParameterType(np.array([4429.929236457418])))
def run(data):
    # Use the model object loaded by init().
    result = model.predict(data)

    # You can return any JSON-serializable object.
    return result.tolist()
