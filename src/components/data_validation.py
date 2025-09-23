import os
import sys
from src.logger import logging
from json import dump as json_dump
from src.exception import MyException
from src.constants import SCHEMA_FILEPATH
from src.entity.config_entity import DataValidationConfig
from src.utils.main_utils import read_yaml_file, read_csv_file, save_as_json
from src.entity.artifact_entity import DataIngestionArtifacts, DataValidationArtifacts


class DataValidation:
    """
    Class to perform data validation on ingested training and test datasets
    by checking feature counts and presence of required features according to schema.
    """

    def __init__(
        self,
        data_ingestion_artifacts: DataIngestionArtifacts,
        data_validation_config: DataValidationConfig,
    ):
        """
        Initialize DataValidation with ingestion artifacts and validation config.

        Args:
            data_ingestion_artifacts (DataIngestionArtifacts): Paths to ingested train and test data.
            data_validation_config (DataValidationConfig): Configurations like report file paths.
        """
        self.data_ingestion_artifacts = data_ingestion_artifacts
        self.data_validation_config = data_validation_config
        self.schema_config = read_yaml_file(filepath=SCHEMA_FILEPATH)

    def _features_count_validate(self, df) -> bool:
        """
        Validate that dataset contains expected number of features.

        Args:
            df (pandas.DataFrame): Dataset to validate.

        Returns:
            bool: True if feature count matches schema, else False.
        """
        expected_num_features = len(self.schema_config.get("features", []))
        actual_num_features = len(df.columns)
        status = actual_num_features == expected_num_features
        return status

    def _features_exist(self, df) -> bool:
        """
        Check that all required numerical and categorical features exist in the dataset.

        Args:
            df (pandas.DataFrame): Dataset to validate.

        Returns:
            bool: True if all required features exist, else False.
        """
        df_features = df.columns

        missing_numerical_features = [
            feature
            for feature in self.schema_config.get("numerical_features", [])
            if feature not in df_features
        ]

        missing_categorical_features = [
            feature
            for feature in self.schema_config.get("categorical_features", [])
            if feature not in df_features
        ]

        all_features_exist = (
            len(missing_numerical_features) == 0
            and len(missing_categorical_features) == 0
        )
        return all_features_exist

    def initiate_data_validation(self) -> DataValidationArtifacts:
        """
        Perform complete data validation on train and test datasets.

        Returns:
            DataValidationArtifacts: Validation status, messages, and report file path.

        Raises:
            MyException: For any errors during validation.
        """
        try:
            data_validation_message = ""

            train_df = read_csv_file(
                filepath=self.data_ingestion_artifacts.train_filepath
            )

            test_df = read_csv_file(
                filepath=self.data_ingestion_artifacts.test_filepath
            )
            logging.info("Training & testing data read.")

            if not self._features_count_validate(train_df):
                msg = "Training data feature count mismatch with schema."
                logging.warning(msg)
                data_validation_message += msg + "\n"
            else:
                logging.info("Training data feature count matches schema.")

            if not self._features_exist(train_df):
                msg = "Training data missing required numerical/categorical features."
                logging.warning(msg)
                data_validation_message += msg + "\n"
            else:
                logging.info("Required features exist in training data.")
            logging.info("Training data validated.")

            if not self._features_count_validate(test_df):
                msg = "Test data feature count mismatch with schema."
                logging.warning(msg)
                data_validation_message += msg + "\n"
            else:
                logging.info("Testing data feature count matches schema.")

            if not self._features_exist(test_df):
                msg = "Test data missing required numerical/categorical features."
                logging.warning(msg)
                data_validation_message += msg + "\n"
            else:
                logging.info("Required features exist in testing data.")
            logging.info("Testing data validated.")

            data_validation_status = len(data_validation_message) == 0
            if data_validation_status:
                logging.info("Data validation passed.")
            else:
                logging.warning("Data validation failed!")

            data_validation_report = {
                "data_validation_status": data_validation_status,
                "data_validation_message": data_validation_message.strip(),
            }

            data_validation_report_dir = os.path.dirname(
                self.data_validation_config.data_validation_reports_filepath
            )
            os.makedirs(data_validation_report_dir, exist_ok=True)

            save_as_json(
                data=data_validation_report,
                filepath=self.data_validation_config.data_validation_reports_filepath,
                indent=4,
            )
            logging.info("Data validation report saved.")

            return DataValidationArtifacts(
                data_validation_status=data_validation_status,
                data_validation_message=data_validation_message.strip(),
                data_validation_report_filepath=self.data_validation_config.data_validation_reports_filepath,
            )

        except Exception as e:
            raise MyException(e, sys) from e
