import os
from src.constants import *
from dataclasses import dataclass, field
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%d-%b-%y_%H:%M:%S")


@dataclass
class TrainingPipelineConfig:
    """
    Data class for configuration of the training pipeline.

    Attributes:
        pipeline_name (str): Name of the pipeline.
        timestamp (str): Timestamp when the pipeline configuration is created.
        artifact_dirpath (str): Directory path to store pipeline artifacts, combined with timestamp.
    """

    pipeline_name: str = field(default=PIPELINE_NAME)
    timestamp: str = field(default=TIMESTAMP)
    artifact_dirpath: str = field(
        default_factory=lambda: os.path.join(ARTIFACT_PATHNAME, TIMESTAMP)
    )


training_pipeline_config = TrainingPipelineConfig()


@dataclass
class DataIngestionConfig:
    """
    Data class for configuration related to data ingestion.

    Attributes:
        data_ingestion_dirpath (str): Base directory path for data ingestion.
        feature_store_dirpath (str): Directory path for feature store.
        data_ingested_dirpath (str): Directory path for ingested data.
        data_filepath (str): File path for fetched data.
        train_data_filepath (str): File path for training data CSV.
        test_data_filepath (str): File path for test data CSV.
        test_size (float): Size of the test dataset.
        collection_name (str): MongoDB collection name.
    """

    data_ingestion_dirpath: str = field(
        default_factory=lambda: os.path.join(
            training_pipeline_config.artifact_dirpath, DATA_INGESTION_DIRNAME
        )
    )
    feature_store_dirpath: str = field(init=False)
    data_filepath: str = field(init=False)
    data_ingested_dirpath: str = field(init=False)
    train_data_filepath: str = field(init=False)
    test_data_filepath: str = field(init=False)
    test_size: float = field(default=TEST_SIZE)
    collection_name: str = field(default=COLLECTION_NAME)

    def __post_init__(self):
        self.feature_store_dirpath = os.path.join(
            self.data_ingestion_dirpath, FEATURE_STORE_DIRNAME
        )
        self.data_filepath = os.path.join(self.feature_store_dirpath, "data.csv")
        self.data_ingested_dirpath = os.path.join(
            self.data_ingestion_dirpath, DATA_INGESTION_INGESTED_DIRNAME
        )
        self.train_data_filepath = os.path.join(
            self.data_ingested_dirpath, TRAIN_DATA_FILENAME
        )
        self.test_data_filepath = os.path.join(
            self.data_ingested_dirpath, TEST_DATA_FILENAME
        )


@dataclass
class DataValidationConfig:
    """
    Data class for configuration related to data validation.

    Attributes:
        data_validation_dirpath (str): Base directory for data validation artifacts.
        data_validation_reports_dirpath (str): Directory for validation reports.
        data_validation_reports_filepath (str): Path for a single consolidated validation report file.
    """

    data_validation_dirpath: str = field(
        default_factory=lambda: os.path.join(
            training_pipeline_config.artifact_dirpath, DATA_VALIDATION_DIRNAME
        )
    )
    data_validation_reports_dirpath: str = field(init=False)
    data_validation_reports_filepath: str = field(init=False)

    def __post_init__(self):
        self.data_validation_reports_dirpath = os.path.join(
            self.data_validation_dirpath, DATA_VALIDATION_REPORT_DIRNAME
        )
        self.data_validation_reports_filepath = os.path.join(
            self.data_validation_reports_dirpath, DATA_VALIDATION_REPORT_FILENAME
        )
