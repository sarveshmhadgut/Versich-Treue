import os
import sys
import numpy as np
from typing import Optional
from src.logger import logging
from src.exception import MyException
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTraining
from src.components.model_evaluation import ModelEvaluation
from src.components.model_deployment import ModelDeployment
from src.cloud_storage.aws_storage import SimpleStorageService

from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainingConfig,
    ModelEvaluationConfig,
    ModelDeploymentConfig,
)
from src.entity.artifact_entity import (
    DataIngestionArtifacts,
    DataValidationArtifacts,
    DataTransformationArtifacts,
    ModelTrainingArtifacts,
    ModelEvaluationArtifacts,
    ModelDeploymentArtifacts,
    ClassificationMetricsArtifacts,
)

terminal_width: int = os.get_terminal_size().columns if os.isatty(1) else 80


class TrainPipeline:
    """
    End-to-end machine learning training pipeline orchestrator.

    This class manages the complete ML pipeline workflow by coordinating different
    components and ensuring proper data flow between pipeline stages. It handles
    data ingestion, validation, transformation, model training, evaluation, and
    deployment in a sequential manner.

    The pipeline follows these stages:
    1. Data Ingestion: Fetch and split raw data
    2. Data Validation: Validate data quality and schema
    3. Data Transformation: Preprocess and transform data
    4. Model Training: Train ML models and compute metrics
    5. Model Evaluation: Compare against existing models
    6. Model Deployment: Push accepted models to production

    Attributes:
        data_ingestion_config (DataIngestionConfig): Configuration for data ingestion stage
        data_validation_config (DataValidationConfig): Configuration for data validation stage
        data_transformation_config (DataTransformationConfig): Configuration for data transformation stage
        model_training_config (ModelTrainingConfig): Configuration for model training stage
        model_evaluation_config (ModelEvaluationConfig): Configuration for model evaluation stage
        model_deployment_config (ModelDeploymentConfig): Configuration for model deployment stage
    """

    def __init__(self) -> None:
        """
        Initialize the TrainPipeline with required configuration objects.

        Sets up all necessary configuration objects for each pipeline stage.
        Each configuration contains paths, parameters, and settings required
        for the respective pipeline component.

        Raises:
            MyException: If any configuration initialization fails
        """
        try:
            self.data_ingestion_config: DataIngestionConfig = DataIngestionConfig()
            self.data_validation_config: DataValidationConfig = DataValidationConfig()
            self.data_transformation_config: DataTransformationConfig = (
                DataTransformationConfig()
            )
            self.model_training_config: ModelTrainingConfig = ModelTrainingConfig()
            self.model_evaluation_config: ModelEvaluationConfig = (
                ModelEvaluationConfig()
            )
            self.model_deployment_config: ModelDeploymentConfig = (
                ModelDeploymentConfig()
            )

        except Exception as e:
            raise MyException(e, sys) from e

    def start_data_ingestion(self) -> DataIngestionArtifacts:
        """
        Execute the data ingestion stage of the pipeline.

        This method initializes the DataIngestion component and executes the
        data ingestion workflow to fetch data from source, split it into
        training and testing datasets, and store them for subsequent stages.

        Returns:
            DataIngestionArtifacts: Contains file paths for ingested train and test datasets

        Raises:
            MyException: If data ingestion process fails
        """
        try:
            data_ingestion: DataIngestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )

            data_ingestion_artifacts: DataIngestionArtifacts = (
                data_ingestion.initiate_data_ingestion()
            )

            return data_ingestion_artifacts

        except Exception as e:
            raise MyException(e, sys) from e

    def start_data_validation(
        self,
        data_ingestion_artifacts: DataIngestionArtifacts,
    ) -> DataValidationArtifacts:
        """
        Execute the data validation stage of the pipeline.

        This method validates the quality, schema, and integrity of the ingested
        datasets. It performs checks for data drift, missing values, and schema
        compliance to ensure data quality before proceeding to transformation.

        Args:
            data_ingestion_artifacts (DataIngestionArtifacts): Artifacts from data ingestion
                containing paths to train and test datasets

        Returns:
            DataValidationArtifacts: Contains validation status, messages, and report paths

        Raises:
            MyException: If data validation process fails
        """
        try:
            data_validation: DataValidation = DataValidation(
                data_ingestion_artifacts=data_ingestion_artifacts,
                data_validation_config=self.data_validation_config,
            )

            data_validation_artifacts: DataValidationArtifacts = (
                data_validation.initiate_data_validation()
            )

            return data_validation_artifacts

        except Exception as e:
            raise MyException(e, sys) from e

    def start_data_transformation(
        self,
        data_ingestion_artifacts: DataIngestionArtifacts,
        data_validation_artifacts: DataValidationArtifacts,
    ) -> DataTransformationArtifacts:
        """
        Execute the data transformation stage of the pipeline.

        This method applies preprocessing transformations to the validated datasets,
        including feature engineering, encoding, scaling, and other transformations
        required to prepare data for model training.

        Args:
            data_ingestion_artifacts (DataIngestionArtifacts): Artifacts from data ingestion
                containing paths to raw datasets
            data_validation_artifacts (DataValidationArtifacts): Artifacts from data validation
                containing validation status and reports

        Returns:
            DataTransformationArtifacts: Contains paths to transformed datasets and
                transformation objects

        Raises:
            MyException: If data transformation process fails
        """
        try:
            data_transformation: DataTransformation = DataTransformation(
                data_ingestion_artifacts=data_ingestion_artifacts,
                data_validation_artifacts=data_validation_artifacts,
                data_transformation_config=self.data_transformation_config,
            )

            data_transformation_artifacts: DataTransformationArtifacts = (
                data_transformation.initiate_data_transformation()
            )

            return data_transformation_artifacts

        except Exception as e:
            raise MyException(e, sys) from e

    def start_model_training(
        self,
        data_transformation_artifacts: DataTransformationArtifacts,
        model_training_config: ModelTrainingConfig,
    ) -> ModelTrainingArtifacts:
        """
        Execute the model training stage of the pipeline.

        This method trains machine learning models using the transformed datasets,
        computes performance metrics, and saves the trained model for evaluation.

        Args:
            data_transformation_artifacts (DataTransformationArtifacts): Artifacts from
                data transformation containing paths to preprocessed datasets
            model_training_config (ModelTrainingConfig): Configuration containing
                hyperparameters and training settings

        Returns:
            ModelTrainingArtifacts: Contains trained model path, metrics, and reports

        Raises:
            MyException: If model training process fails
        """
        try:
            model_trainer: ModelTraining = ModelTraining(
                data_transformation_artifacts=data_transformation_artifacts,
                model_training_config=model_training_config,
            )

            model_training_artifacts: ModelTrainingArtifacts = (
                model_trainer.initiate_model_training()
            )

            return model_training_artifacts

        except Exception as e:
            raise MyException(e, sys) from e

    def start_model_evaluation(
        self,
        data_ingestion_artifacts: DataIngestionArtifacts,
        model_training_artifacts: ModelTrainingArtifacts,
        data_transformation_artifacts: DataTransformationArtifacts,
    ) -> ModelEvaluationArtifacts:
        """
        Execute the model evaluation stage of the pipeline.

        This method compares the newly trained model against the best existing model
        (if available) from production storage to determine if the new model should
        be accepted for deployment.

        Args:
            data_ingestion_artifacts (DataIngestionArtifacts): Artifacts containing
                test dataset paths for evaluation
            model_training_artifacts (ModelTrainingArtifacts): Artifacts containing
                the newly trained model and its metrics

        Returns:
            ModelEvaluationArtifacts: Contains evaluation results and deployment decision

        Raises:
            MyException: If model evaluation process fails
        """
        try:
            model_evaluation: ModelEvaluation = ModelEvaluation(
                data_ingestion_artifacts=data_ingestion_artifacts,
                data_transformation_artifacts=data_transformation_artifacts,
                model_training_artifacts=model_training_artifacts,
                model_evaluation_config=self.model_evaluation_config,
            )

            model_evaluation_artifacts: ModelEvaluationArtifacts = (
                model_evaluation.initiate_model_evaluation()
            )

            return model_evaluation_artifacts

        except Exception as e:
            raise MyException(e, sys) from e

    def start_model_deployment(
        self, model_evaluation_artifacts: ModelEvaluationArtifacts
    ) -> ModelDeploymentArtifacts:
        """
        Execute the model deployment stage of the pipeline.

        This method pushes the accepted model to production storage (S3) for
        serving and inference. Only models that pass the evaluation criteria
        are deployed to production.

        Args:
            model_evaluation_artifacts (ModelEvaluationArtifacts): Artifacts containing
                evaluation results and model paths

        Returns:
            ModelDeploymentArtifacts: Contains deployment confirmation and storage details

        Raises:
            MyException: If model deployment process fails
        """
        try:
            model_deployment: ModelDeployment = ModelDeployment(
                model_evaluation_artifacts=model_evaluation_artifacts,
                model_deployment_config=self.model_deployment_config,
                s3=SimpleStorageService(),
            )

            model_deployment_artifacts: ModelDeploymentArtifacts = (
                model_deployment.initiate_model_deployment()
            )

            return model_deployment_artifacts

        except Exception as e:
            raise MyException(e, sys) from e

    def run_pipeline(self) -> Optional[ModelDeploymentArtifacts]:
        """
        Execute the complete end-to-end training pipeline workflow.

        This method orchestrates the entire ML pipeline by sequentially executing
        all stages: data ingestion, validation, transformation, model training,
        evaluation, and deployment. The pipeline includes decision logic to only
        deploy models that meet acceptance criteria.

        Pipeline Flow:
        1. Data Ingestion: Fetch and split raw data
        2. Data Validation: Validate data quality and schema
        3. Data Transformation: Preprocess and feature engineer
        4. Model Training: Train models and compute metrics
        5. Model Evaluation: Compare against production models
        6. Model Deployment: Deploy accepted models to production

        Returns:
            Optional[ModelDeploymentArtifacts]: Deployment artifacts if model is accepted
                and deployed, None if model is rejected

        Raises:
            MyException: If any pipeline stage fails during execution
        """
        try:
            print("=" * terminal_width)
            logging.info("Excecuting training pipeline...")
            print("-" * terminal_width)

            # Stage 1: Data Ingestion
            logging.info("Executing Data Ingestion...")
            data_ingestion_artifacts: DataIngestionArtifacts = (
                self.start_data_ingestion()
            )
            logging.info("Data Ingestion completed")
            print(data_ingestion_artifacts)
            print("-" * terminal_width)

            # Stage 2: Data Validation
            logging.info("Executing Data Validation...")
            data_validation_artifacts: DataValidationArtifacts = (
                self.start_data_validation(
                    data_ingestion_artifacts=data_ingestion_artifacts
                )
            )
            logging.info("Data Validation completed")
            print(data_validation_artifacts)
            print("-" * terminal_width)

            # Stage 3: Data Transformation
            logging.info("Executing Data Transformation...")
            data_transformation_artifacts: DataTransformationArtifacts = (
                self.start_data_transformation(
                    data_ingestion_artifacts=data_ingestion_artifacts,
                    data_validation_artifacts=data_validation_artifacts,
                )
            )
            logging.info("Data Transformation completed")
            print(data_transformation_artifacts)
            print("-" * terminal_width)

            # Stage 4: Model Training
            logging.info("Executing Model Training...")
            model_training_artifacts: ModelTrainingArtifacts = (
                self.start_model_training(
                    data_transformation_artifacts=data_transformation_artifacts,
                    model_training_config=self.model_training_config,
                )
            )
            logging.info("Model Training completed")
            print(model_training_artifacts)
            print("-" * terminal_width)

            # Stage 5: Model Evaluation (using latest known artifact run paths)
            logging.info("Executing Model Evaluation...")
            model_evaluation_artifacts: ModelEvaluationArtifacts = (
                self.start_model_evaluation(
                    data_ingestion_artifacts=data_ingestion_artifacts,
                    data_transformation_artifacts=data_transformation_artifacts,
                    model_training_artifacts=model_training_artifacts,
                )
            )
            logging.info("Model Evaluation completed")
            print(model_evaluation_artifacts)
            print("-" * terminal_width)

            if not model_evaluation_artifacts.model_acceptance:
                logging.warning(
                    f"Model rejected for deployment. "
                    f"Accuracy discrepancy: {model_evaluation_artifacts.accuracy_discrepancy}"
                )
                logging.info("Training pipeline completed")
                print("=" * terminal_width)
                return None

            # Stage 6: Model Deployment (only if accepted)
            logging.info("Executing Model Deployment...")
            model_deployment_artifacts: ModelDeploymentArtifacts = (
                self.start_model_deployment(
                    model_evaluation_artifacts=model_evaluation_artifacts
                )
            )

            logging.info("Model Deployment completed")
            print(model_deployment_artifacts)
            print("-" * terminal_width)

            logging.info("Training pipeline completed")
            print("=" * terminal_width)
            return model_deployment_artifacts

        except Exception as e:
            logging.exception("Training pipeline execution failed")
            print("=" * terminal_width)
            print("Training pipeline failed!")
            print("=" * terminal_width)
            raise MyException(e, sys) from e
