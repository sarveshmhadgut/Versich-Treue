import os
from typing import Any
from src.constants import *
from dataclasses import dataclass, field
from src.utils.main_utils import get_current_timestamp, read_yaml_file


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
    timestamp: str = field(default_factory=get_current_timestamp)
    artifact_dirpath: str = field(
        default_factory=lambda: os.path.join(ARTIFACT_PATHNAME, get_current_timestamp())
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
            self.data_validation_dirpath, REPORT_DIRNAME
        )
        self.data_validation_reports_filepath = os.path.join(
            self.data_validation_reports_dirpath, REPORT_FILENAME
        )


@dataclass
class DataTransformationConfig:
    """
    Data class for configuration related to data transformation.

    Attributes:
        data_transformation_dirpath (str): Base directory for data transformation artifacts.
        data_transformation_transformed_data_dirpath (str): Directory for transformed data numpy arrays.
        data_transformation_object_filepath (str): File path for the serialized transformation object.
        data_transformation_train_array_filepath (str): File path for transformed training dataset numpy array.
        data_transformation_test_array_filepath (str): File path for transformed test dataset numpy array.
    """

    data_transformation_dirpath: str = field(
        default_factory=lambda: os.path.join(
            training_pipeline_config.artifact_dirpath, DATA_TRANSFORMATION_DIRNAME
        )
    )
    data_transformation_transformed_data_dirpath: str = field(init=False)
    data_transformation_object_dirpath: str = field(init=False)
    data_transformation_object_filepath: str = field(init=False)
    data_transformation_train_array_filepath: str = field(init=False)
    data_transformation_test_array_filepath: str = field(init=False)

    def __post_init__(self):
        self.data_transformation_transformed_data_dirpath = os.path.join(
            self.data_transformation_dirpath,
            DATA_TRANSFORMATION_TRANSFORMED_DATA_DIRPATH,
        )

        self.data_transformation_object_dirpath = os.path.join(
            self.data_transformation_dirpath, DATA_TRANSFORMATION_OBJECT_DIRNAME
        )

        self.data_transformation_object_filepath = os.path.join(
            self.data_transformation_object_dirpath,
            DATA_TRANSFORMATION_OBJECT_FILENAME,
        )

        self.data_transformation_train_array_filepath = os.path.join(
            self.data_transformation_transformed_data_dirpath,
            TRAIN_DATA_FILENAME.replace("csv", "npy"),
        )

        self.data_transformation_test_array_filepath = os.path.join(
            self.data_transformation_transformed_data_dirpath,
            TEST_DATA_FILENAME.replace("csv", "npy"),
        )


@dataclass
class ModelTrainingConfig:
    """
    Data class for configuration related to model training.

    Attributes:
        model_training_dirpath (str): Base directory for model training artifacts.
        trained_model_dirpath (str): Directory where trained model is stored.
        trained_model_filepath (str): File path for the serialized trained model.
        report_filepath (str): File path for the classification metrics report.
        threshold_accuracy (float): Minimum accuracy threshold for model acceptance.
        training_model_params (dict): Dictionary containing all model hyperparameters.
    """

    model_training_dirpath: str = field(
        default_factory=lambda: os.path.join(
            training_pipeline_config.artifact_dirpath, MODEL_TRAINING_DIRNAME
        )
    )

    trained_model_dirpath: str = field(init=False)
    trained_model_filepath: str = field(init=False)
    report_filepath: str = field(init=False)
    threshold_accuracy: float = 0.5
    training_model_params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.trained_model_dirpath = os.path.join(
            self.model_training_dirpath, TRAINED_MODEL_DIRNAME
        )

        self.report_dirpath = os.path.join(self.model_training_dirpath, REPORT_DIRNAME)

        self.trained_model_filepath = os.path.join(
            self.trained_model_dirpath, MODEL_FILENAME
        )

        self.report_filepath = os.path.join(self.report_dirpath, REPORT_FILENAME)

        self.model_params = read_yaml_file(MODEL_PARAMS_FILEPATH) or {}
        self.threshold_accuracy = self.model_params.get(
            "threshold_accuracy", self.threshold_accuracy
        )

        self.training_model_params = {
            "bootstrap": self.model_params.get("bootstrap", True),
            "class_weight": self.model_params.get("class_weight", None),
            "criterion": self.model_params.get("criterion", "gini"),
            "max_depth": self.model_params.get("max_depth", None),
            "max_features": self.model_params.get("max_features", "auto"),
            "max_leaf_nodes": self.model_params.get("max_leaf_nodes", None),
            "max_samples": self.model_params.get("max_samples", None),
            "min_samples_split": self.model_params.get("min_samples_split", 2),
            "min_samples_leaf": self.model_params.get("min_samples_leaf", 1),
            "min_weight_fraction_leaf": self.model_params.get(
                "min_weight_fraction_leaf", 0.0
            ),
            "n_estimators": self.model_params.get("n_estimators", 100),
            "oob_score": self.model_params.get("oob_score", False),
            "random_state": self.model_params.get("random_state", 42),
        }
