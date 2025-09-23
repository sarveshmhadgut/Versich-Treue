import os
import sys
from typing import Optional
from src.logger import logging
from src.exception import MyException
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTraining

from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainingConfig,
)
from src.entity.artifact_entity import (
    DataIngestionArtifacts,
    DataValidationArtifacts,
    DataTransformationArtifacts,
    ModelTrainingArtifacts,
)


terminal_width: int = os.get_terminal_size().columns if os.isatty(1) else 80


class TrainPipeline:
    """
    Pipeline class to orchestrate the complete training workflow including
    data ingestion, validation, preprocessing, model training, and evaluation.

    This class manages the end-to-end machine learning pipeline by coordinating
    different components and ensuring proper data flow between pipeline stages.
    """

    def __init__(self) -> None:
        """
        Initialize the TrainPipeline with required configuration objects.

        Sets up data ingestion, validation, and transformation configurations needed for
        the complete pipeline execution.

        Raises:
            MyException: If configuration initialization fails.
        """
        try:
            self.data_ingestion_config: DataIngestionConfig = DataIngestionConfig()
            self.data_validation_config: DataValidationConfig = DataValidationConfig()
            self.data_transformation_config: DataTransformationConfig = (
                DataTransformationConfig()
            )
            self.model_training_config: ModelTrainingConfig = ModelTrainingConfig()

        except Exception as e:
            raise MyException(e, sys) from e

    def start_data_ingestion(self) -> DataIngestionArtifacts:
        """
        Start the data ingestion process and return the resulting artifacts.

        This method initializes the DataIngestion component and executes the
        data ingestion workflow to fetch, split, and store training/test datasets.

        Returns:
            DataIngestionArtifacts: Paths for the ingested train and test datasets.

        Raises:
            MyException: If data ingestion fails.
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
        Start the data validation process using artifacts from data ingestion.

        Args:
            data_ingestion_artifacts (DataIngestionArtifacts): Paths to ingested datasets.

        Returns:
            DataValidationArtifacts: Validation status and reports.

        Raises:
            MyException: If data validation fails.
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
        Start the data transformation process.

        Args:
            data_ingestion_artifacts (DataIngestionArtifacts): Paths to ingested datasets.
            data_validation_artifacts (DataValidationArtifacts): Data validation results.

        Returns:
            DataTransformationArtifacts: Paths to transformation artifacts.

        Raises:
            MyException: If data transformation fails.
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
        Start the model training process using transformed data.

        Args:
            data_transformation_artifacts (DataTransformationArtifacts): Paths to transformed datasets.
            model_training_config (ModelTrainingConfig): Configuration for model training.

        Returns:
            ModelTrainingArtifacts: Paths and metrics from model training.

        Raises:
            MyException: If model training fails.
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

    def run_pipeline(self) -> None:
        """
        Execute the complete training pipeline workflow.

        This method orchestrates the entire pipeline by sequentially executing:
        1. Data ingestion
        2. Data validation
        3. Data transformation
        4. Model training

        Raises:
            MyException: On any failure during pipeline execution.
        """
        try:
            print("=" * terminal_width)
            logging.info("Initializing training pipeline...")
            print("-" * terminal_width)

            # Step 1: Data Ingestion
            logging.info("Initializing data ingestion pipeline...")
            data_ingestion_artifacts: DataIngestionArtifacts = (
                self.start_data_ingestion()
            )
            logging.info("Data ingestion pipeline completed.")
            print("-" * terminal_width)

            # Step 2: Data Validation
            logging.info("Initializing data validation pipeline...")
            data_validation_artifacts: DataValidationArtifacts = (
                self.start_data_validation(
                    data_ingestion_artifacts=data_ingestion_artifacts
                )
            )
            logging.info("Data validation pipeline completed.")
            print("-" * terminal_width)

            # Step 3: Data Transformation
            logging.info("Initializing data transformation pipeline...")
            data_transformation_artifacts: DataTransformationArtifacts = (
                self.start_data_transformation(
                    data_ingestion_artifacts=data_ingestion_artifacts,
                    data_validation_artifacts=data_validation_artifacts,
                )
            )

            logging.info("Data transformation pipeline completed.")
            print("-" * terminal_width)

            # Step 4: Model Training
            logging.info("Initializing model training pipeline...")
            model_training_artifacts: ModelTrainingArtifacts = (
                self.start_model_training(
                    data_transformation_artifacts=data_transformation_artifacts,
                    model_training_config=self.model_training_config,
                )
            )
            logging.info("Model training pipeline completed.")
            print("-" * terminal_width)
            # Model evaluation,
            # Model deployment

            logging.info("Training pipeline completed.")
            print("=" * terminal_width)

        except Exception as e:
            raise MyException(e, sys) from e
