import os
from src.constants import *
from typing import Dict, Any
from dataclasses import dataclass, field
from src.utils.main_utils import get_current_timestamp, read_yaml_file


@dataclass
class TrainingPipelineConfig:
    """
    Configuration for the main training pipeline.

    This class defines the core pipeline configuration including naming,
    timestamps, and artifact storage paths.

    Attributes:
        pipeline_name (str): Name identifier for the pipeline
        timestamp (str): Timestamp when the pipeline configuration is created
        artifact_dirpath (str): Directory path to store pipeline artifacts with timestamp
    """

    pipeline_name: str = field(default=PIPELINE_NAME)
    timestamp: str = field(default_factory=get_current_timestamp)
    artifact_dirpath: str = field(
        default_factory=lambda: os.path.join(ARTIFACT_PATHNAME, get_current_timestamp())
    )


# Global pipeline configuration instance
training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()


@dataclass
class DataIngestionConfig:
    """
    Configuration for the data ingestion stage of the pipeline.

    This class manages file paths and parameters for data ingestion including
    data fetching, train-test splitting, and MongoDB collection access.

    Attributes:
        data_filepath (str): File path for the complete fetched dataset
        train_data_filepath (str): File path for training data CSV
        test_data_filepath (str): File path for test data CSV
        test_size (float): Proportion of dataset to use for testing (0.0 to 1.0)
        collection_name (str): MongoDB collection name for data source
    """

    data_filepath: str = field(init=False)
    train_data_filepath: str = field(init=False)
    test_data_filepath: str = field(init=False)
    test_size: float = field(default=TEST_SIZE)
    collection_name: str = field(default=COLLECTION_NAME)

    def __post_init__(self) -> None:
        """
        Initialize computed file paths after dataclass creation.

        This method constructs the complete file paths for data storage
        based on the pipeline configuration and predefined directory structure.
        """
        self.data_filepath = os.path.join(
            training_pipeline_config.artifact_dirpath,
            DATA_INGESTION_DIRNAME,
            FEATURE_STORE_DIRNAME,
            "data.csv",
        )

        self.train_data_filepath = os.path.join(
            training_pipeline_config.artifact_dirpath,
            DATA_INGESTION_DIRNAME,
            DATA_INGESTION_INGESTED_DIRNAME,
            TRAIN_DATA_FILENAME,
        )

        self.test_data_filepath = os.path.join(
            training_pipeline_config.artifact_dirpath,
            DATA_INGESTION_DIRNAME,
            DATA_INGESTION_INGESTED_DIRNAME,
            TEST_DATA_FILENAME,
        )


@dataclass
class DataValidationConfig:
    """
    Configuration for the data validation stage of the pipeline.

    This class manages paths and parameters for data quality validation,
    schema validation, and drift detection processes.

    Attributes:
        data_validation_reports_filepath (str): Path for consolidated validation report file
    """

    data_validation_reports_filepath: str = field(init=False)

    def __post_init__(self) -> None:
        """
        Initialize validation report file path after dataclass creation.

        Constructs the complete file path for storing data validation reports
        based on the pipeline configuration.
        """
        self.data_validation_reports_filepath = os.path.join(
            training_pipeline_config.artifact_dirpath,
            DATA_VALIDATION_DIRNAME,
            REPORT_DIRNAME,
            REPORT_FILENAME,
        )


@dataclass
class DataTransformationConfig:
    """
    Configuration for the data transformation stage of the pipeline.

    This class manages paths for data preprocessing, feature engineering,
    and transformation object serialization.

    Attributes:
        data_transformation_object_filepath (str): Path for serialized transformation object
        data_transformation_train_array_filepath (str): Path for transformed training numpy array
        data_transformation_test_array_filepath (str): Path for transformed test numpy array
    """

    data_transformation_object_filepath: str = field(init=False)
    data_transformation_train_array_filepath: str = field(init=False)
    data_transformation_test_array_filepath: str = field(init=False)

    def __post_init__(self) -> None:
        """
        Initialize transformation file paths after dataclass creation.

        Constructs complete file paths for storing transformation objects
        and processed numpy arrays based on the pipeline configuration.
        """
        self.data_transformation_object_filepath = os.path.join(
            training_pipeline_config.artifact_dirpath,
            DATA_TRANSFORMATION_DIRNAME,
            DATA_TRANSFORMATION_OBJECT_DIRNAME,
            DATA_TRANSFORMATION_OBJECT_FILENAME,
        )

        self.data_transformation_train_array_filepath = os.path.join(
            training_pipeline_config.artifact_dirpath,
            DATA_TRANSFORMATION_DIRNAME,
            DATA_TRANSFORMATION_TRANSFORMED_DATA_DIRPATH,
            TRAIN_DATA_FILENAME.replace("csv", "npy"),
        )

        self.data_transformation_test_array_filepath = os.path.join(
            training_pipeline_config.artifact_dirpath,
            DATA_TRANSFORMATION_DIRNAME,
            DATA_TRANSFORMATION_TRANSFORMED_DATA_DIRPATH,
            TEST_DATA_FILENAME.replace("csv", "npy"),
        )


@dataclass
class ModelTrainingConfig:
    """
    Configuration for the model training stage of the pipeline.

    This class manages parameters for model training including hyperparameters,
    file paths for model artifacts, and performance thresholds.

    Attributes:
        trained_model_filepath (str): Path for serialized trained model
        report_filepath (str): Path for classification metrics report
        threshold_accuracy (float): Minimum accuracy threshold for model acceptance
        training_model_params (Dict[str, Any]): Dictionary of model hyperparameters
    """

    trained_model_filepath: str = field(init=False)
    report_filepath: str = field(init=False)
    threshold_accuracy: float = field(default=0.5)
    training_model_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Initialize model training paths and parameters after dataclass creation.

        Loads model parameters from YAML configuration and constructs file paths
        for model storage and reporting based on the pipeline configuration.
        """
        # Initialize file paths
        self.trained_model_filepath = os.path.join(
            training_pipeline_config.artifact_dirpath,
            MODEL_TRAINING_DIRNAME,
            TRAINED_MODEL_DIRNAME,
            MODEL_FILENAME,
        )

        self.report_filepath = os.path.join(
            training_pipeline_config.artifact_dirpath,
            MODEL_TRAINING_DIRNAME,
            REPORT_DIRNAME,
            REPORT_FILENAME,
        )

        # Load model parameters from YAML file
        model_params: Dict[str, Any] = read_yaml_file(MODEL_PARAMS_FILEPATH) or {}

        # Update threshold accuracy from parameters
        self.threshold_accuracy = model_params.get(
            "threshold_accuracy", self.threshold_accuracy
        )

        # Set RandomForest hyperparameters with defaults
        self.training_model_params = {
            "bootstrap": model_params.get("bootstrap", True),
            "class_weight": model_params.get("class_weight", None),
            "criterion": model_params.get("criterion", "gini"),
            "max_depth": model_params.get("max_depth", None),
            "max_features": model_params.get("max_features", "sqrt"),
            "max_leaf_nodes": model_params.get("max_leaf_nodes", None),
            "max_samples": model_params.get("max_samples", None),
            "min_samples_split": model_params.get("min_samples_split", 2),
            "min_samples_leaf": model_params.get("min_samples_leaf", 1),
            "min_weight_fraction_leaf": model_params.get(
                "min_weight_fraction_leaf", 0.0
            ),
            "n_estimators": model_params.get("n_estimators", 100),
            "oob_score": model_params.get("oob_score", False),
            "random_state": model_params.get("random_state", 42),
        }


@dataclass
class ModelEvaluationConfig:
    """
    Configuration for the model evaluation stage of the pipeline.

    This class manages parameters for comparing newly trained models
    against previously deployed models in S3 storage.

    Attributes:
        model_evaluation_threshold_score (float): Minimum score threshold for model acceptance
        model_evaluation_report_filepath (str): Path for evaluation report file
        bucket_name (str): S3 bucket name for model storage
        s3_model_key_path (str): S3 object key path for the best model
    """

    model_evaluation_threshold_score: float = field(init=False)
    model_evaluation_report_filepath: str = field(init=False)
    bucket_name: str = field(init=False)
    s3_model_key_path: str = field(init=False)

    def __post_init__(self) -> None:
        """
        Initialize model evaluation configuration after dataclass creation.

        Sets up evaluation thresholds, report paths, and S3 storage configuration
        based on predefined constants and pipeline configuration.
        """
        self.model_evaluation_threshold_score = MODEL_EVALUATION_THRESHOLD

        self.model_evaluation_report_filepath = os.path.join(
            training_pipeline_config.artifact_dirpath,
            MODEL_EVALUATION_DIRNAME,
            REPORT_DIRNAME,
            REPORT_FILENAME,
        )

        self.bucket_name = MODEL_BUCKET_NAME
        self.s3_model_key_path = MODEL_FILENAME


@dataclass
class ModelDeploymentConfig:
    """
    Configuration for the model deployment stage of the pipeline.

    This class manages parameters for pushing trained models to production
    storage in S3 for serving and inference.

    Attributes:
        bucket_name (str): S3 bucket name for model deployment
        s3_model_key_path (str): S3 object key path for deployed model
    """

    bucket_name: str = field(default=MODEL_BUCKET_NAME)
    s3_model_key_path: str = field(default=MODEL_FILENAME)


@dataclass
class OwnerClassifierConfig:
    """
    Configuration for the owner classifier model serving and inference.

    This class manages configuration parameters for loading and using
    the trained model for prediction tasks, including model file paths
    and S3 storage settings for model retrieval.

    Attributes:
        model_filepath (str): File path or name for the trained model file
        model_bucket_name (str): S3 bucket name containing the deployed model
    """

    model_filepath: str = field(default=MODEL_FILENAME)
    model_bucket_name: str = field(default=MODEL_BUCKET_NAME)
