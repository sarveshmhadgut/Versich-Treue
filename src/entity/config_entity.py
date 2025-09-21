import os
from src.constants import *
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

TIMESTAMP = datetime.now().strftime("%d-%b-%y_%H:%M:%S")


@dataclass
class TrainingPipelineConfig:
    """
    Data class for configuration of the training pipeline.

    Attributes:
        pipeline_name (str): Name of the pipeline.
        timestamp (str): Timestamp when the pipeline configuration is created.
        artifact_dirpath (str): Directory path to store pipeline artifacts, combined with timestamp.
    """

    pipeline_name: str = PIPELINE_NAME
    timestamp: str = TIMESTAMP
    artifact_dirpath: str = os.path.join(ARTIFACT_PATHNAME, timestamp)


training_pipeline_config = TrainingPipelineConfig()


@dataclass
class DataIngestionConfig:
    """
    Data class for configuration related to data ingestion.

    Attributes:
        data_ingestion_dirpath (str): Base directory path for data ingestion.
        feature_store_dirpath (str): Directory path for feature store.
        data_ingested_dirpath (str): Directory path for ingested data.
        data_filepath (str): File path for fetched data
        train_data_filepath (str): File path for training data CSV.
        test_data_filepath (str): File path for test data CSV.
        test_size (float): Size of the test dataset.
        collection_name (str): MongoDB collection name.
    """

    data_ingestion_dirpath: str = os.path.join(
        training_pipeline_config.artifact_dirpath, DATA_INGESTION_DIRNAME
    )
    feature_store_dirpath: str = os.path.join(
        data_ingestion_dirpath, FEATURE_STORE_DIRNAME
    )
    data_filepath: str = os.path.join(feature_store_dirpath, "data.csv")

    data_ingested_dirpath: str = os.path.join(
        data_ingestion_dirpath, DATA_INGESTION_INGESTED_DIRNAME
    )
    train_data_filepath: str = os.path.join(data_ingested_dirpath, TRAIN_DATA_FILENAME)
    test_data_filepath: str = os.path.join(data_ingested_dirpath, TEST_DATA_FILENAME)

    test_size: float = TEST_SIZE
    collection_name: str = COLLECTION_NAME
