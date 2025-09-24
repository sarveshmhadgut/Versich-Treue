import os
import sys
from pandas import Series, DataFrame
from src.exception import MyException
from sklearn.base import BaseEstimator
from typing import Any, Optional, Union
from src.configuration.aws_connection import S3
from src.cloud_storage.aws_storage import SimpleStorageService


class S3Estimator:
    """
    A machine learning model estimator that integrates with Amazon S3 for model storage and retrieval.

    This class provides functionality to load, save, and use machine learning models stored in S3,
    with built-in prediction capabilities and automatic model caching.

    Attributes:
        bucket_name (str): The name of the S3 bucket containing the model
        model_filepath (str): The S3 key path to the model file
        s3 (SimpleStorageService): The S3 service instance for cloud operations
        fetched_model (Optional[Any]): Cached model instance loaded from S3
    """

    def __init__(self, bucket_name: str, model_filepath: str) -> None:
        """
        Initialize the S3 estimator with bucket and model file information.

        Args:
            bucket_name (str): The name of the S3 bucket containing the model
            model_filepath (str): The S3 key path to the model file

        Raises:
            MyException: If S3 service initialization fails
        """
        try:
            self.bucket_name: str = bucket_name
            self.model_filepath: str = model_filepath
            self.s3: SimpleStorageService = SimpleStorageService()
            self.fetched_model: Optional[Any] = None

        except Exception as e:
            raise MyException(e, sys) from e

    def s3_model_found(self, model_filepath: Optional[str] = None) -> bool:
        """
        Check if the model file exists in the S3 bucket.

        Args:
            model_filepath (Optional[str]): Optional custom model filepath to check. If None, uses the instance's model_filepath.

        Returns:
            bool: True if the model file exists in S3, False otherwise

        Raises:
            MyException: If S3 operation fails
        """
        try:
            filepath = model_filepath or self.model_filepath
            exists = self.s3.key_path_exists(
                bucket_name=self.bucket_name, s3_key=filepath
            )
            return exists

        except Exception as e:
            raise MyException(e, sys) from e

    def load_model(self) -> Any:
        """
        Load the machine learning model from S3.

        Returns:
            Any: The loaded machine learning model object

        Raises:
            MyException: If model loading from S3 fails
            MyException: If model file does not exist in S3
        """
        try:
            if not self.s3_model_found():
                raise MyException(e, sys)

            if "/" in self.model_filepath:
                model_dirpath = os.path.dirname(self.model_filepath)
                model_filename = os.path.basename(self.model_filepath)

            else:
                model_dirpath = None
                model_filename = self.model_filepath

            model = self.s3.load_model(
                model_filename=model_filename,
                model_dirpath=model_dirpath,
                bucket_name=self.bucket_name,
            )

            return model

        except Exception as e:
            raise MyException(e, sys) from e

    def save_model(self, from_filename: str, remove: bool = True) -> None:
        """
        Save a model file to S3.

        Args:
            from_filename (str): Local filepath of the model to upload
            remove (bool): Whether to remove the local file after upload. Defaults to True

        Raises:
            MyException: If model upload to S3 fails
            FileNotFoundError: If local model file does not exist
        """
        try:

            if not os.path.exists(from_filename):
                raise FileNotFoundError(
                    f"Local model file '{from_filename}' does not exist"
                )

            self.s3.upload_file(
                from_filename=from_filename,
                to_filename=self.model_filepath,
                bucket_name=self.bucket_name,
                remove=remove,
            )

        except FileNotFoundError as e:
            raise MyException(e, sys) from e

        except Exception as e:
            raise MyException(e, sys) from e

    def predict(self, df: DataFrame) -> Union[Series, DataFrame]:
        """
        Make predictions using the S3-stored model.

        Automatically loads the model from S3 if not already cached,
        then uses it to make predictions on the provided data.

        Args:
            df (pd.DataFrame): Input data for making predictions

        Returns:
            Union[pd.Series, pd.DataFrame]: Model predictions

        Raises:
            MyException: If model loading or prediction fails
            ValueError: If input DataFrame is empty or invalid
        """
        try:
            if df.empty:
                raise ValueError("Input DataFrame cannot be empty")

            if not self.fetched_model:
                self.fetched_model = self.load_model()

            if not hasattr(self.fetched_model, "predict"):
                raise AttributeError(
                    f"Loaded model of type '{type(self.fetched_model)}' does not have a 'predict' method"
                )

            predictions = self.fetched_model.predict(df)
            return predictions

        except (ValueError, AttributeError) as e:
            raise MyException(e, sys) from e

        except Exception as e:
            raise MyException(e, sys) from e
