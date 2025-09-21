import sys
from src.logger import logging
from src.exception import MyException
from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifacts


class TrainPipeline:
    """
    Pipeline class to orchestrate the training data ingestion and further steps.
    """

    def __init__(self) -> None:
        """
        Initialize the TrainPipeline with the data ingestion configuration.
        """
        self.data_ingestion_config: DataIngestionConfig = DataIngestionConfig()
        logging.info("Train Pipeline initialized.")

    def start_data_ingestion(self) -> DataIngestionArtifacts:
        """
        Start the data ingestion process and return the resulting artifacts.

        Returns:
            DataIngestionArtifacts: Contains filepaths for the ingested train and test data.

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
            logging.info("Data ingestion pipeline completed successfully.")
            return data_ingestion_artifacts

        except Exception as e:
            raise MyException(e, sys)

    def run_pipeline(self) -> None:
        """
        Run the training pipeline starting with data ingestion.
        Extend this method to include model training and evaluation steps.
        """
        try:
            self.start_data_ingestion()
            logging.info("Training pipeline executed successfully.")

        except Exception as e:
            raise MyException(e, sys)
