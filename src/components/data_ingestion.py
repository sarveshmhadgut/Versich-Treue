import os
import sys
from pandas import DataFrame
from src.logger import logging
from src.exception import MyException
from src.data_access.vt_data import VTData
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
            logging.info("Data ingestion configured successfully.")

        except Exception as e:
            raise MyException(e, sys) from e

    def export_data_to_feature_store(self) -> DataFrame:
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

            os.makedirs(self.data_ingestion_config.feature_store_dirpath, exist_ok=True)
            feature_store_path = os.path.join(
                self.data_ingestion_config.feature_store_dirpath, "data.csv"
            )

            df.to_csv(feature_store_path, index=False)
            logging.info(f"Feature store data fetched and stored successfully")

            return df

        except Exception as e:
            raise MyException(e, sys) from e

    def train_test_splitting(self, df: DataFrame) -> None:
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
            os.makedirs(self.data_ingestion_config.data_ingested_dirpath, exist_ok=True)

            train_path: str = self.data_ingestion_config.train_data_filepath
            test_path: str = self.data_ingestion_config.test_data_filepath

            train_data.to_csv(train_path, index=False, header=True)
            test_data.to_csv(test_path, index=False, header=True)

            logging.info(f"Training and testing data created successfully")

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
            df: DataFrame = self.export_data_to_feature_store()
            if df is None or df.empty:
                raise MyException("Exported dataframe is empty")

            self.train_test_splitting(df=df)

            data_ingestion_artifacts = DataIngestionArtifacts(
                train_filepath=self.data_ingestion_config.train_data_filepath,
                test_filepath=self.data_ingestion_config.test_data_filepath,
            )
            logging.info("Data ingestion artifacts created successfully.")
            return data_ingestion_artifacts

        except Exception as e:
            raise MyException(e, sys) from e
