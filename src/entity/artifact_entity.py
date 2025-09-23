from dataclasses import dataclass

@dataclass
class DataIngestionArtifacts:
    """
    Data class that encapsulates the file paths generated as a result of
    the data ingestion process, typically splitting raw data into
    training and testing subsets.

    Attributes:
        train_filepath (str): Path where the ingested training dataset CSV file is stored.
        test_filepath (str): Path where the ingested test dataset CSV file is stored.
    """

    train_filepath: str
    test_filepath: str


@dataclass
class DataValidationArtifacts:
    """
    Data class that stores the outcomes of the data validation process,
    including the status, descriptive messages, and reports.

    Attributes:
        data_validation_status (bool): Indicates whether validation passed (True) or failed (False).
        data_validation_message (str): Human-readable message providing details about validation results.
        data_validation_report_filepath (str): Path to the generated validation report file.
    """

    data_validation_status: bool
    data_validation_message: str
    data_validation_report_filepath: str


@dataclass
class DataTransformationArtifacts:
    """
    Data class that encapsulates the file paths generated as a result of
    the data transformation process.

    Attributes:
        data_transformation_object_filepath (str): Path where the transformation object file is saved.
        data_transformation_train_filepath (str): Path where the transformed training data numpy array is saved.
        data_transformation_test_filepath (str): Path where the transformed test data numpy array is saved.
    """

    data_transformation_object_filepath: str
    data_transformation_train_filepath: str
    data_transformation_test_filepath: str
