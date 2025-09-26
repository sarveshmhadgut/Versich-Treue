import sys
import numpy as np
from halo import Halo
from typing import Optional
from src.logger import logging
from dataclasses import dataclass
from sklearn.metrics import accuracy_score
from src.exception import MyException
from src.constants import SCHEMA_FILEPATH
from src.entity.s3_estimator import S3Estimator
from src.entity.config_entity import ModelEvaluationConfig
from src.utils.main_utils import (
    save_as_json,
    read_yaml_file,
    load_numpy_array,
)
from src.entity.artifact_entity import (
    DataIngestionArtifacts,
    DataTransformationArtifacts,
    ModelTrainingArtifacts,
    ModelEvaluationArtifacts,
)


@dataclass
class EvaluateModelResponse:
    """
    Response data class containing model evaluation metrics and comparison results.

    Attributes:
        trained_model_f1_score (float): F1 score of the newly trained model
        fetched_model_f1_score (float): F1 score of the model fetched from S3
        model_acceptance (bool): Whether the newly trained model should be accepted
        accuracy_discrepancy (float): Difference between trained and fetched model scores
    """

    model_acceptance: bool
    trained_model_f1_score: float
    fetched_model_f1_score: float
    accuracy_discrepancy: float


class ModelEvaluation:
    """
    Model evaluation component for comparing newly trained models against deployed models.

    This class handles the evaluation process by comparing a newly trained model's performance
    against the best model currently deployed in S3 storage. It applies necessary data
    preprocessing transformations and computes evaluation metrics to determine whether
    the new model should be accepted for deployment.

    Attributes:
        data_ingestion_artifacts (DataIngestionArtifacts): Artifacts from data ingestion stage
        model_training_artifacts (ModelTrainingArtifacts): Artifacts from model training stage
        model_evaluation_config (ModelEvaluationConfig): Configuration for model evaluation
        schema_config (dict): Schema configuration loaded from YAML file
    """

    def __init__(
        self,
        data_ingestion_artifacts: DataIngestionArtifacts,
        data_transformation_artifacts: DataTransformationArtifacts,
        model_training_artifacts: ModelTrainingArtifacts,
        model_evaluation_config: ModelEvaluationConfig,
    ) -> None:
        """
        Initialize the ModelEvaluation component.

        Args:
            data_ingestion_artifacts (DataIngestionArtifacts): Data ingestion stage artifacts
            model_training_artifacts (ModelTrainingArtifacts): Model training stage artifacts
            model_evaluation_config (ModelEvaluationConfig): Model evaluation configuration

        Raises:
            MyException: If initialization fails or schema file cannot be read
        """
        try:

            self.data_ingestion_artifacts = data_ingestion_artifacts
            self.data_transformation_artifacts = data_transformation_artifacts
            self.model_training_artifacts = model_training_artifacts
            self.model_evaluation_config = model_evaluation_config
            self.schema_config = read_yaml_file(SCHEMA_FILEPATH)

        except Exception as e:
            raise MyException(e, sys) from e

    def fetch_best_model(self) -> Optional[S3Estimator]:
        """
        Fetch the best model from S3 storage if it exists.

        This method attempts to retrieve the currently deployed model from S3 storage
        for comparison against the newly trained model.

        Returns:
            Optional[S3Estimator]: S3 estimator instance if model exists, None otherwise

        Raises:
            MyException: If S3 model fetching fails
        """
        try:

            bucket_name = self.model_evaluation_config.bucket_name
            model_key_path = self.model_evaluation_config.s3_model_key_path

            s3_estimator = S3Estimator(
                bucket_name=bucket_name, model_filepath=model_key_path
            )

            if s3_estimator.s3_model_found():
                return s3_estimator

            else:
                return None

        except Exception as e:
            raise MyException(e, sys) from e

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Evaluate the newly trained model against the best model from S3.

        This method loads test data, applies preprocessing transformations,
        and computes evaluation metrics for both the newly trained model
        and the model fetched from S3 (if available).

        Returns:
            EvaluateModelResponse: Evaluation results with metrics and comparison

        Raises:
            MyException: If model evaluation process fails
        """
        try:
            trained_model_accuracy = (
                self.model_training_artifacts.classification_metrics_artifacts.accuracy
            )
            test_df = load_numpy_array(
                self.data_transformation_artifacts.data_transformation_test_array_filepath
            )

            X_test = test_df[:, :-1]
            y_test = test_df[:, -1]

            s3_model_accuracy = None
            s3_model = self.fetch_best_model()

            if s3_model:
                logging.info("Fetched best model from S3")

                with Halo(text="Computing F1 score...", spinner="dots"):
                    y_hat = s3_model.predict(X=X_test)
                    s3_model_accuracy = accuracy_score(y_true=y_test, y_pred=y_hat)

            s3_model_accuracy = (
                0.0 if not s3_model_accuracy else round(s3_model_accuracy, 5)
            )

            accuracy_discrepancy = float(trained_model_accuracy - s3_model_accuracy)
            logging.info(f"F1-score Discrepancy: {round(accuracy_discrepancy, 5)}")

            model_acceptance = bool(trained_model_accuracy > s3_model_accuracy)
            if model_acceptance:
                logging.info(f"Trained model accepted")

            else:
                logging.info(f"Trained model rejected")

            model_evaluation_report = {
                "Model accepted": bool(model_acceptance),
                "Trained model F1 score": round(float(s3_model_accuracy), 5),
                "Fetched model F1 score": round(float(s3_model_accuracy), 5),
                "Accuracy discrepancy": round(float(accuracy_discrepancy), 5),
            }

            save_as_json(
                data=model_evaluation_report,
                filepath=self.model_evaluation_config.model_evaluation_report_filepath,
                indent=4,
            )

            model_evaluation_response = EvaluateModelResponse(
                model_acceptance=model_acceptance,
                trained_model_f1_score=trained_model_accuracy,
                fetched_model_f1_score=s3_model_accuracy,
                accuracy_discrepancy=accuracy_discrepancy,
            )

            return model_evaluation_response

        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifacts:
        """
        Initiate the complete model evaluation process.

        This method orchestrates the entire model evaluation workflow and
        creates the evaluation artifacts for the next pipeline stage.

        Returns:
            ModelEvaluationArtifacts: Artifacts containing evaluation results

        Raises:
            MyException: If model evaluation initiation fails
        """
        try:
            model_evaluation_response = self.evaluate_model()

            s3_model_path = self.model_evaluation_config.s3_model_key_path

            model_evaluation_artifacts = ModelEvaluationArtifacts(
                model_acceptance=model_evaluation_response.model_acceptance,
                trained_model_path=self.model_training_artifacts.trained_model_filepath,
                s3_model_path=s3_model_path,
                accuracy_discrepancy=model_evaluation_response.accuracy_discrepancy,
            )

            return model_evaluation_artifacts

        except Exception as e:
            raise MyException(e, sys) from e
