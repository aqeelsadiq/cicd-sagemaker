import boto3
from sagemaker.sklearn.model import SKLearnModel
from sagemaker import Session
import time

endpoint_name = "iris-endpoint"
role = "arn:aws:iam::387867038403:role/sagemaker-mlops"
s3_model_path = "s3://my-mlops-project-aq/model/model.tar.gz"

sagemaker_session = Session()
sm_client = boto3.client("sagemaker")

# ------------------------------
# 1️⃣ Delete endpoint if exists
# ------------------------------
try:
    resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
    print("Endpoint exists. Deleting endpoint...")
    sm_client.delete_endpoint(EndpointName=endpoint_name)

    # Wait for deletion
    while True:
        resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = resp["EndpointStatus"]
        print("Waiting for endpoint deletion... Status:", status)
        if status == "Deleting":
            time.sleep(10)
        else:
            break
except sm_client.exceptions.ClientError:
    print("Endpoint does not exist. Continuing...")

# ------------------------------
# 2️⃣ Delete endpoint config if exists
# ------------------------------
try:
    sm_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
    print("Deleted existing endpoint config.")
except sm_client.exceptions.ClientError:
    print("Endpoint config does not exist. Continuing...")

# ------------------------------
# 3️⃣ Create model
# ------------------------------
model = SKLearnModel(
    model_data=s3_model_path,
    role=role,
    entry_point="inference/inference.py",
    framework_version="1.2-1",  # Use compatible sklearn version
    sagemaker_session=sagemaker_session
)

# ------------------------------
# 4️⃣ Deploy endpoint
# ------------------------------
predictor = model.deploy(
    instance_type="ml.t2.medium",
    initial_instance_count=1,
    endpoint_name=endpoint_name
)

print("Model deployed at endpoint:", predictor.endpoint_name)
