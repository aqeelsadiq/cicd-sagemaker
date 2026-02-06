# predict.py

import boto3
import json

# -----------------------------
# 1️⃣ Create SageMaker runtime client
# -----------------------------
client = boto3.client("sagemaker-runtime", region_name="us-east-1")  # Replace with your AWS region

# -----------------------------
# 2️⃣ Define input data
# Must match features order: sepal_length, sepal_width, petal_length, petal_width
# -----------------------------
payload = [5.1, 3.5, 1.4, 0.2]  # Example single row
# For multiple rows, you can use: payload = [[5.1,3.5,1.4,0.2],[6.2,2.8,4.8,1.8]]

# Convert to JSON
input_data = json.dumps(payload)

# -----------------------------
# 3️⃣ Invoke SageMaker endpoint
# -----------------------------
response = client.invoke_endpoint(
    EndpointName="iris-endpoint",      # Replace with your endpoint name
    Body=input_data,
    ContentType="application/json"
)

# -----------------------------
# 4️⃣ Parse prediction
# -----------------------------
result = response['Body'].read().decode("utf-8")
prediction = json.loads(result)  # This will return a list like [0]

print("Prediction:", prediction)

# Optional: decode numeric label back to species name if you saved LabelEncoder
species_map = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}
print("Predicted species:", species_map[prediction[0]])

#this below is working and tested
# import boto3
# import json

# endpoint_name = "iris-endpoint"
# client = boto3.client('sagemaker-runtime')

# # Input must be JSON list
# payload = json.dumps([5.1, 3.5, 1.4, 0.2])

# response = client.invoke_endpoint(
#     EndpointName=endpoint_name,
#     Body=payload,
#     ContentType="application/json"
# )

# result = response['Body'].read().decode()
# print("Prediction:", result)
