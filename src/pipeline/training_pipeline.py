import os
import sys
from src.logger import logging
from src.exception import MyException
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation

from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
)
from src.entity.artifact_entity import (
    DataIngestionArtifacts,
    DataValidationArtifacts,
    DataTransformationArtifacts,
)


terminal_width = os.get_terminal_size().columns if os.isatty(1) else 80


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
        """
        try:
            self.data_ingestion_config: DataIngestionConfig = DataIngestionConfig()
            self.data_validation_config: DataValidationConfig = DataValidationConfig()
            self.data_transformation_config: DataTransformationConfig = (
                DataTransformationConfig()
            )
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
            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )

            data_ingestion_artifacts = data_ingestion.initiate_data_ingestion()
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
            data_validation = DataValidation(
                data_ingestion_artifacts=data_ingestion_artifacts,
                data_validation_config=self.data_validation_config,
            )

            data_validation_artifacts = data_validation.initiate_data_validation()
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
            data_transformation = DataTransformation(
                data_ingestion_artifacts=data_ingestion_artifacts,
                data_validation_artifacts=data_validation_artifacts,
                data_transformation_config=self.data_transformation_config,
            )

            data_transformation_artifacts = (
                data_transformation.initiate_data_transformation()
            )
            return data_transformation_artifacts

        except Exception as e:
            raise MyException(e, sys) from e

    def run_pipeline(self) -> None:
        """
        Execute the complete training pipeline workflow.

        This method orchestrates the entire pipeline by sequentially executing:
        1. Data ingestion
        2. Data validation
        3. Data transformation
        # Future: Model training, evaluation, and deployment.

        Raises:
            MyException: On any failure during pipeline execution.
        """
        try:
            print("=" * terminal_width)
            logging.info("Starting training pipeline...")
            print()
            # Step 1: Data Ingestion
            logging.info("Starting data ingestion pipeline...")
            data_ingestion_artifacts = self.start_data_ingestion()
            logging.info("Data ingestion pipeline completed.")
            print("-" * terminal_width)

            # Step 2: Data Validation
            logging.info("Starting data validation pipeline...")
            data_validation_artifacts = self.start_data_validation(
                data_ingestion_artifacts=data_ingestion_artifacts
            )
            logging.info("Data validation pipeline completed.")
            print("-" * terminal_width)

            # Step 3: Data Transformation
            logging.info("Starting data transformation pipeline...")
            data_transformation_artifacts = self.start_data_transformation(
                data_ingestion_artifacts=data_ingestion_artifacts,
                data_validation_artifacts=data_validation_artifacts,
            )
            logging.info("Data transformation pipeline completed.")
            print("-" * terminal_width)

            # Model training,
            # Model evaluation,
            # Model deployment

            logging.info("Training pipeline completed.")
            print("=" * terminal_width)
        except Exception as e:
            raise MyException(e, sys) from e
