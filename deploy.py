import boto3
import time
from sagemaker.sklearn import SKLearnModel
from sagemaker import Session

endpoint_name = "iris-endpoint"
role = "arn:aws:iam::387867038403:role/sagemaker-mlops"
s3_model_path = "s3://my-mlops-project-aq/model/model.tar.gz"

sagemaker_session = Session()
client = boto3.client("sagemaker")

# 1️⃣ Delete endpoint if it exists
try:
    client.describe_endpoint(EndpointName=endpoint_name)
    print("Endpoint exists. Deleting endpoint...")
    client.delete_endpoint(EndpointName=endpoint_name)
    client.delete_endpoint_config(EndpointConfigName=endpoint_name)
    # Wait until endpoint is deleted
    while True:
        try:
            status = client.describe_endpoint(EndpointName=endpoint_name)['EndpointStatus']
            print("Waiting for endpoint deletion, current status:", status)
            time.sleep(10)
        except client.exceptions.ClientError:
            break
    print("Endpoint deleted.")
except client.exceptions.ClientError:
    print("Endpoint does not exist. Continuing...")

# 2️⃣ Deploy new model
model = SKLearnModel(
    model_data=s3_model_path,
    role=role,
    entry_point="inference/inference.py",
    framework_version="1.2-1",  # use compatible version
    sagemaker_session=sagemaker_session
)

predictor = model.deploy(
    instance_type="ml.t2.medium",
    initial_instance_count=1,
    endpoint_name=endpoint_name
)

print("Model deployed at endpoint:", predictor.endpoint_name)


# deploy.py
#this below is working
# import sagemaker
# from sagemaker.sklearn.model import SKLearnModel
# from sagemaker import Session

# # Replace with your S3 model path
# s3_model_path = "s3://my-mlops-project-aq/model/model.tar.gz"

# # Replace with your SageMaker execution role ARN
# role = "arn:aws:iam::387867038403:role/sagemaker-mlops"

# # SageMaker session
# sagemaker_session = Session()

# # Create SKLearn model
# model = SKLearnModel(
#     model_data=s3_model_path,
#     role=role,
#     entry_point="inference/inference.py",
#     framework_version="1.2-1",  # use 1.2-1 for sklearn v1.2
#     sagemaker_session=sagemaker_session
# )

# # Deploy endpoint
# predictor = model.deploy(
#     instance_type="ml.t2.medium",
#     initial_instance_count=1,
#     endpoint_name="iris-endpoint",
#     update_endpoint=True
# )

# print("Model deployed at endpoint:", predictor.endpoint_name)
