from dataclasses import dataclass


@dataclass
class DataIngestionArtifacts:
    """
    Data class to store file paths resulting from data ingestion process.

    Attributes:
        test_filepath (str): File path to the test dataset CSV file.
        train_filepath (str): File path to the training dataset CSV file.
    """

    test_filepath: str
    train_filepath: str
