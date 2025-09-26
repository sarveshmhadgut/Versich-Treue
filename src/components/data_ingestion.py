import os
import sys
from pandas import DataFrame
from src.logger import logging
from src.exception import MyException
from src.data_access.vt_data import VTData
from src.utils.main_utils import save_df_as_csv
from sklearn.model_selection import train_test_split
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifacts


class DataIngestion:
    """
    Handles data ingestion tasks including export from MongoDB,
    storing in feature store, and train/test splitting.
    """

    def __init__(
        self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()
    ) -> None:
        """
        Initialize DataIngestion with configuration for directories and file paths.

        Args:
            data_ingestion_config (DataIngestionConfig): Configuration object for data ingestion.

        Raises:
            MyException: For any initialization failures.
        """
        try:
            self.data_ingestion_config: DataIngestionConfig = data_ingestion_config

        except Exception as e:
            raise MyException(e, sys) from e

    def _export_data_to_feature_store(self) -> DataFrame:
        """
        Export data from MongoDB collection to a CSV file in the feature store.

        Returns:
            DataFrame: The dataframe containing exported data.

        Raises:
            MyException: If there is an error in fetching or saving data.
        """
        try:
            data: VTData = VTData()
            df: DataFrame = data.export_collection_as_dataframe(
                collection_name=self.data_ingestion_config.collection_name
            )

            feature_store_dir = os.path.dirname(
                self.data_ingestion_config.data_filepath
            )
            os.makedirs(feature_store_dir, exist_ok=True)
            save_df_as_csv(
                df=df, filepath=self.data_ingestion_config.data_filepath, index=False
            )

            return df

        except Exception as e:
            raise MyException(e, sys) from e

    def _train_test_splitting(self, df: DataFrame) -> None:
        """
        Split the dataframe into train and test datasets and save them as CSV files.

        Args:
            df (DataFrame): The dataframe to split.

        Raises:
            MyException: If there is an error during train/test splitting or saving.
        """
        try:
            train_data, test_data = train_test_split(
                df, test_size=self.data_ingestion_config.test_size, random_state=42
            )
            ingested_dir = os.path.dirname(
                self.data_ingestion_config.train_data_filepath
            )
            os.makedirs(ingested_dir, exist_ok=True)

            train_path: str = self.data_ingestion_config.train_data_filepath
            test_path: str = self.data_ingestion_config.test_data_filepath

            save_df_as_csv(df=train_data, filepath=train_path, index=False, header=True)
            save_df_as_csv(df=test_data, filepath=test_path, index=False, header=True)

        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        """
        Orchestrate the data ingestion process: export data and split it.

        Returns:
            DataIngestionArtifacts: Artifact object containing train and test filepaths.

        Raises:
            MyException: If any step in the ingestion process fails.
        """
        try:
            df: DataFrame = self._export_data_to_feature_store()

            if df is None or df.empty:
                error_msg = "Exported dataframe is empty."
                logging.error(error_msg)
                raise MyException(error_msg)
            logging.info(f"Exported data to feature store with shape {df.shape}.")

            self._train_test_splitting(df=df)
            logging.info("Train-test splitting completed.")

            data_ingestion_artifacts = DataIngestionArtifacts(
                train_filepath=self.data_ingestion_config.train_data_filepath,
                test_filepath=self.data_ingestion_config.test_data_filepath,
            )

            logging.info("Data ingestion process completed.")
            return data_ingestion_artifacts

        except Exception as e:
            raise MyException(e, sys) from e
