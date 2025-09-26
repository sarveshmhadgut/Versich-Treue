import sys
from typing import Type
from src.logger import logging
from src.exception import MyException
from src.entity.s3_estimator import S3Estimator
from src.entity.config_entity import ModelDeploymentConfig
from src.cloud_storage.aws_storage import SimpleStorageService
from src.entity.artifact_entity import (
    ModelEvaluationArtifacts,
    ModelDeploymentArtifacts,
)


class ModelDeployment:
    """
    Model deployment component for pushing trained models to production storage.

    This class handles the deployment of accepted machine learning models to AWS S3
    storage, making them available for serving, inference, and future model comparisons.
    It coordinates with S3 storage services to upload model artifacts and create
    deployment records.

    Attributes:
        model_evaluation_artifacts (ModelEvaluationArtifacts): Artifacts from model evaluation stage
        model_deployment_config (ModelDeploymentConfig): Configuration for model deployment
        s3 (SimpleStorageService): S3 storage service instance for cloud operations
        s3_estimator (S3Estimator): S3 estimator instance for model operations
    """

    def __init__(
        self,
        model_evaluation_artifacts: ModelEvaluationArtifacts,
        model_deployment_config: ModelDeploymentConfig,
        s3: SimpleStorageService,
        s3_estimator: Type[S3Estimator] = S3Estimator,
    ) -> None:
        """
        Initialize the ModelDeployment component.

        Sets up the deployment environment with evaluation artifacts, configuration,
        and S3 services required for model deployment operations.

        Args:
            model_evaluation_artifacts (ModelEvaluationArtifacts): Artifacts containing
                evaluation results and model paths from the evaluation stage
            model_deployment_config (ModelDeploymentConfig): Configuration containing
                S3 bucket details and deployment parameters
            s3 (SimpleStorageService): Instance of S3 storage service for cloud operations
            s3_estimator (Type[S3Estimator]): S3Estimator class for creating model estimator instance.
                Defaults to S3Estimator

        Raises:
            MyException: If initialization fails or S3 configuration is invalid
        """
        try:

            self.model_evaluation_artifacts = model_evaluation_artifacts
            self.model_deployment_config = model_deployment_config
            self.s3 = s3

            self.s3_estimator = s3_estimator(
                bucket_name=self.model_deployment_config.bucket_name,
                model_filepath=self.model_deployment_config.s3_model_key_path,
            )

        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_deployment(self) -> ModelDeploymentArtifacts:
        """
        Execute the model deployment process to production storage.

        This method uploads the trained model from local storage to S3 bucket,
        making it available for serving and future model comparisons. The deployment
        only proceeds if the model has been accepted during the evaluation stage.

        Returns:
            ModelDeploymentArtifacts: Contains deployment confirmation with bucket name
                and S3 model path

        Raises:
            MyException: If model deployment fails or S3 upload encounters errors

        Note:
            This method assumes the model has already passed evaluation criteria.
            It performs the actual file transfer to production storage.
        """
        try:
            if not self.model_evaluation_artifacts.trained_model_path:
                raise ValueError(
                    "No trained model path provided in evaluation artifacts"
                )

            self.s3_estimator.save_model(
                from_filename=self.model_evaluation_artifacts.trained_model_path
            )
            logging.info("Model uploaded to S3")

            model_deployment_artifacts = ModelDeploymentArtifacts(
                bucket_name=self.model_deployment_config.bucket_name,
                s3_model_path=self.model_deployment_config.s3_model_key_path,
            )

            logging.info(
                f"Model deployed to: s3://{model_deployment_artifacts.bucket_name}/"
                f"{model_deployment_artifacts.s3_model_path}"
            )

            return model_deployment_artifacts

        except ValueError as e:
            raise MyException(e, sys) from e

        except Exception as e:
            raise MyException(e, sys) from e
