# # deploy.py

# import sagemaker
# from sagemaker.sklearn.model import SKLearnModel
# from sagemaker import Session

# # Replace with your S3 path
# s3_model_path = "s3://my-mlops-project-aq/model/model.tar.gz"

# # Replace with your SageMaker execution role ARN
# role = "arn:aws:iam::387867038403:role/sagemaker-mlops"

# # SageMaker session
# sagemaker_session = Session()

# # Create SageMaker Model
# model = SKLearnModel(
#     model_data=s3_model_path,
#     role=role,
#     entry_point="inference/inference.py",
#     framework_version="1.2-1",  # compatible with SDK v2
#     sagemaker_session=sagemaker_session
# )

# # Deploy endpoint
# predictor = model.deploy(
#     instance_type="ml.t2.medium",
#     initial_instance_count=1,
#     endpoint_name="iris-endpoint"
# )

# print("Model deployed at endpoint:", predictor.endpoint_name)

# deploy.py

import sagemaker
from sagemaker.sklearn.model import SKLearnModel
from sagemaker import Session

# Replace with your S3 model path
s3_model_path = "s3://my-mlops-project-aq/model/model.tar.gz"

# Replace with your SageMaker execution role ARN
role = "arn:aws:iam::387867038403:role/sagemaker-mlops"

# SageMaker session
sagemaker_session = Session()

# Create SKLearn model
model = SKLearnModel(
    model_data=s3_model_path,
    role=role,
    entry_point="inference/inference.py",
    framework_version="1.2-1",  # use 1.2-1 for sklearn v1.2
    sagemaker_session=sagemaker_session
)

# Deploy endpoint
predictor = model.deploy(
    instance_type="ml.t2.medium",
    initial_instance_count=1,
    endpoint_name="iris-endpoint",
    update_endpoint=True
)

print("Model deployed at endpoint:", predictor.endpoint_name)
