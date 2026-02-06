# # inference/inference.py

# import os
# import joblib
# import numpy as np
# import json

# def model_fn(model_dir):
#     """Load model from the directory"""
#     model_path = os.path.join(model_dir, "model.pkl")
#     return joblib.load(model_path)

# def input_fn(request_body, content_type='application/json'):
#     """Parse input data"""
#     if content_type == 'application/json':
#         # Expecting input as a list: [5.1, 3.5, 1.4, 0.2]
#         data = json.loads(request_body)
#         return np.array(data).reshape(1, -1)
#     raise ValueError(f"Unsupported content type: {content_type}")

# def predict_fn(input_data, model):
#     """Make prediction"""
#     return model.predict(input_data)

# def output_fn(prediction, content_type='application/json'):
#     """Format prediction output"""
#     if content_type == 'application/json':
#         return json.dumps(prediction.tolist())
#     raise ValueError(f"Unsupported content type: {content_type}")

# inference/inference.py

import os
import joblib
import numpy as np
import json

def model_fn(model_dir):
    """Load model from the directory"""
    model_path = os.path.join(model_dir, "model.pkl")
    return joblib.load(model_path)

def input_fn(request_body, content_type='application/json'):
    """Parse input data"""
    if content_type == 'application/json':
        # Expecting input as a list: [5.1, 3.5, 1.4, 0.2]
        data = json.loads(request_body)
        return np.array(data).reshape(1, -1)
    raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    """Make prediction"""
    return model.predict(input_data)

def output_fn(prediction, content_type='application/json'):
    """Format prediction output"""
    if content_type == 'application/json':
        return json.dumps(prediction.tolist())
    raise ValueError(f"Unsupported content type: {content_type}")
