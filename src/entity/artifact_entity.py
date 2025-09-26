from dataclasses import dataclass


@dataclass
class DataIngestionArtifacts:
    """
    Data class that encapsulates the file paths generated as a result of
    the data ingestion process, typically splitting raw data into
    training and testing subsets.

    This artifact is produced by the data ingestion component and consumed
    by subsequent pipeline stages that require access to the split datasets.

    Attributes:
        train_filepath (str): Path where the ingested training dataset CSV file is stored
        test_filepath (str): Path where the ingested test dataset CSV file is stored
    """

    train_filepath: str
    test_filepath: str


@dataclass
class DataValidationArtifacts:
    """
    Data class that stores the outcomes of the data validation process,
    including the status, descriptive messages, and reports.

    This artifact provides feedback about data quality, schema compliance,
    and data drift detection results to guide pipeline decision-making.

    Attributes:
        data_validation_status (bool): Indicates whether validation passed (True) or failed (False)
        data_validation_message (str): Human-readable message providing details about validation results
        data_validation_report_filepath (str): Path to the generated validation report file
    """

    data_validation_status: bool
    data_validation_message: str
    data_validation_report_filepath: str


@dataclass
class DataTransformationArtifacts:
    """
    Data class that encapsulates the file paths generated as a result of
    the data transformation process.

    This artifact contains paths to the preprocessed data arrays and the
    transformation object that can be used for inference on new data.

    Attributes:
        data_transformation_object_filepath (str): Path where the transformation object file is saved
        data_transformation_train_array_filepath (str): Path where the transformed training data numpy array is saved
        data_transformation_test_array_filepath (str): Path where the transformed test data numpy array is saved
    """

    data_transformation_object_filepath: str
    data_transformation_train_array_filepath: str
    data_transformation_test_array_filepath: str


@dataclass
class ClassificationMetricsArtifacts:
    """
    Data class that stores classification performance metrics calculated
    during model evaluation.

    This artifact provides comprehensive performance metrics that help
    assess model quality and make decisions about model acceptance.

    Attributes:
        accuracy (float): Accuracy score of the classification model (0.0 to 1.0)
        precision (float): Precision score of the classification model (0.0 to 1.0)
        recall (float): Recall score of the classification model (0.0 to 1.0)
        log_loss_ (float): Logarithmic loss of the classification model (lower is better)
        f1_score_ (float): F1 score of the classification model (0.0 to 1.0)
        roc_auc (float): ROC AUC score of the classification model (0.0 to 1.0)
    """

    accuracy: float
    precision: float
    recall: float
    log_loss_: float
    f1_score_: float
    roc_auc: float


@dataclass
class ModelTrainingArtifacts:
    """
    Data class that encapsulates the outputs generated as a result of
    the model training process.

    This artifact contains the trained model file path, performance metrics,
    and reports that are used by subsequent evaluation and deployment stages.

    Attributes:
        trained_model_filepath (str): Path where the trained model object is saved
        report_filepath (str): Path where the classification metrics report is saved
        classification_metrics_artifacts (ClassificationMetricsArtifacts): Object containing calculated classification metrics
    """

    trained_model_filepath: str
    report_filepath: str
    classification_metrics_artifacts: ClassificationMetricsArtifacts


@dataclass
class ModelEvaluationArtifacts:
    """
    Data class that stores the results of model evaluation and comparison
    against previously deployed models.

    This artifact provides decision-making information about whether a newly
    trained model should be accepted for deployment based on performance
    comparison with existing production models.

    Attributes:
        model_acceptance (bool): Whether the newly trained model should be accepted for deployment
        trained_model_path (str): Local file path of the newly trained model
        s3_model_path (str): S3 path where the best model is stored
        accuracy_discrepancy (float): Performance difference between new and existing models
    """

    model_acceptance: bool
    trained_model_path: str
    s3_model_path: str
    accuracy_discrepancy: float


@dataclass
class ModelDeploymentArtifacts:
    """
    Data class that stores the results of model deployment to production storage.

    This artifact confirms successful model deployment and provides location
    information for the deployed model in cloud storage.

    Attributes:
        bucket_name (str): Name of the S3 bucket where the model is deployed
        s3_model_path (str): S3 object key path of the deployed model
    """

    bucket_name: str
    s3_model_path: str
