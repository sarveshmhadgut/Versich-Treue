import os
import sys
from typing import Optional
from src.logger import logging
from src.exception import MyException
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation

from src.entity.config_entity import DataIngestionConfig, DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifacts, DataValidationArtifacts

terminal_width = os.get_terminal_size().columns if os.isatty(1) else 80


class TrainPipeline:
    """
    Pipeline class to orchestrate the complete training workflow including
    data ingestion, validation, preprocessing, model training and evaluation.

    This class manages the end-to-end machine learning pipeline by coordinating
    different components and ensuring proper data flow between pipeline stages.
    """

    def __init__(self) -> None:
        """
        Initialize the TrainPipeline with required configuration objects.

        Sets up data ingestion and validation configurations needed for
        the complete pipeline execution.
        """
        try:
            self.data_ingestion_config: DataIngestionConfig = DataIngestionConfig()
            self.data_validation_config: DataValidationConfig = DataValidationConfig()
        except Exception as e:
            raise MyException(e, sys) from e

    def start_data_ingestion(self) -> DataIngestionArtifacts:
        """
        Start the data ingestion process and return the resulting artifacts.

        This method initializes the DataIngestion component and executes the
        data ingestion workflow to fetch, split and store training/test datasets.

        Returns:
            DataIngestionArtifacts: Contains filepaths for the ingested train and test data.

        Raises:
            MyException: If data ingestion process fails during execution.
        """
        try:
            logging.info("Starting data ingestion pipeline...")

            data_ingestion: DataIngestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )

            data_ingestion_artifacts: DataIngestionArtifacts = (
                data_ingestion.initiate_data_ingestion()
            )

            logging.info("Data ingestion pipeline completed successfully.")
            return data_ingestion_artifacts

        except Exception as e:
            raise MyException(e, sys) from e

    def start_data_validation(
        self,
        data_ingestion_artifacts: DataIngestionArtifacts,
    ) -> DataValidationArtifacts:
        """
        Start the data validation process using artifacts from data ingestion.

        This method validates the quality, schema, and statistical properties
        of the ingested data to ensure it meets the requirements for model training.

        Args:
            data_ingestion_artifacts (DataIngestionArtifacts): Artifacts containing
                paths to ingested train and test datasets.

        Returns:
            DataValidationArtifacts: Contains validation status, message and report filepath.

        Raises:
            MyException: If data validation process fails during execution.
        """
        try:
            print()
            logging.info("Starting data validation pipeline...")

            data_validation: DataValidation = DataValidation(
                data_ingestion_artifacts=data_ingestion_artifacts,
                data_validation_config=self.data_validation_config,
            )

            data_validation_artifacts: DataValidationArtifacts = (
                data_validation.initiate_data_validation()
            )

            logging.info("Data validation pipeline completed successfully.")
            return data_validation_artifacts

        except Exception as e:
            raise MyException(e, sys) from e

    def run_pipeline(self) -> None:
        """
        Execute the complete training pipeline workflow.

        This method orchestrates the entire pipeline by sequentially executing:
        1. Data ingestion to fetch and split datasets
        2. Data validation to ensure data quality
        3. Future steps will include data transformation, model training and evaluation

        The method ensures proper error handling and logging throughout the pipeline.

        Raises:
            MyException: If any stage of the pipeline fails during execution.
        """
        try:
            print("-" * terminal_width)
            logging.info("Starting training pipeline...")

            # Step 1: Data Ingestion
            data_ingestion_artifacts: DataIngestionArtifacts = (
                self.start_data_ingestion()
            )
            print("-" * terminal_width)
            # Step 2: Data Validation
            data_validation_artifacts: DataValidationArtifacts = (
                self.start_data_validation(
                    data_ingestion_artifacts=data_ingestion_artifacts
                )
            )
            print("-" * terminal_width)

            # Future pipeline steps will be added here:
            # - Data transformation
            # - Model training
            # - Model evaluation
            # - Model deployment

            logging.info("Training pipeline completed successfully.")
        except Exception as e:
            raise MyException(e, sys) from e
