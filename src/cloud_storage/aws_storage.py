import os
import sys
import pickle
import pandas as pd
from io import StringIO
from src.logger import logging
from typing import Union, List, Any
from src.exception import MyException
from mypy_boto3_s3.client import S3Client
from src.configuration.aws_connection import S3
from mypy_boto3_s3.service_resource import Bucket, Object
from mypy_boto3_s3.service_resource import S3ServiceResource
from src.utils.main_utils import save_df_as_csv, read_csv_file
from botocore.exceptions import ClientError, NoCredentialsError


class SimpleStorageService:
    """
    A wrapper class for AWS S3 operations providing simplified methods for common tasks.

    This class provides high-level methods for interacting with S3 buckets and objects,
    including file upload/download, CSV handling, and model persistence operations.

    Attributes:
        resource (S3ServiceResource): The S3 service resource client
        client (S3Client): The S3 client for low-level operations
    """

    def __init__(self) -> None:
        """
        Initialize the S3 service wrapper.

        Raises:
            NoCredentialsError: If AWS credentials are not configured
            MyException: If S3 connection cannot be established
        """
        try:
            s3_client: S3 = S3()
            self.resource: S3ServiceResource = s3_client.resource
            self.client: S3Client = s3_client.client

        except NoCredentialsError as e:
            raise MyException(e, sys) from e

        except Exception as e:
            raise MyException(e, sys) from e

    def get_bucket(self, bucket_name: str) -> Bucket:
        """
        Get a bucket resource object.

        Args:
            bucket_name (str): The name of the S3 bucket

        Returns:
            Bucket: S3 bucket resource object

        Raises:
            MyException: If bucket cannot be accessed
        """
        try:
            bucket = self.resource.Bucket(bucket_name)
            return bucket

        except Exception as e:
            raise MyException(e, sys) from e

    def key_path_exists(self, bucket_name: str, s3_key: str) -> bool:
        """
        Check if a key path exists in the specified S3 bucket.

        Args:
            bucket_name (str): The name of the S3 bucket
            s3_key (str): The S3 object key path to check

        Returns:
            bool: True if the key path exists, False otherwise

        Raises:
            MyException: If bucket access fails or operation encounters an error
        """
        try:
            bucket = self.get_bucket(bucket_name=bucket_name)
            file_objects = list(bucket.objects.filter(Prefix=s3_key))

            exists = len(file_objects) > 0
            return exists

        except Exception as e:
            raise MyException(e, sys) from e

    @staticmethod
    def read_object(
        object_name: Object, decode: bool = True, make_readable: bool = False
    ) -> Union[str, bytes, StringIO]:
        """
        Read content from an S3 object.

        Args:
            object_name (Object): The S3 object to read from
            decode (bool): Whether to decode the content as UTF-8 text. Defaults to True
            make_readable (bool): Whether to wrap the content in StringIO for file-like operations. Defaults to False

        Returns:
            Union[str, bytes, StringIO]: The object content in the requested format

        Raises:
            MyException: If reading the object fails
        """
        try:
            content = object_name.get()["Body"].read()

            if decode:
                content = content.decode("utf-8")

            if make_readable and decode:
                return StringIO(content)

            return content

        except Exception as e:
            raise MyException(e, sys) from e

    def get_file_object(
        self, filename: str, bucket_name: str
    ) -> Union[Object, List[Object]]:
        """
        Get file object(s) from S3 bucket matching the filename prefix.

        Args:
            filename (str): The filename or prefix to search for
            bucket_name (str): The name of the S3 bucket

        Returns:
            Union[Object, List[Object]]: Single object if exactly one match, list of objects otherwise

        Raises:
            MyException: If bucket access fails or no objects are found
        """
        try:
            bucket = self.get_bucket(bucket_name=bucket_name)
            file_objects = [
                file_object for file_object in bucket.objects.filter(Prefix=filename)
            ]
            func = lambda x: x[0] if len(x) == 1 else x
            file_objs = func(file_objects)

            return file_objs

        except Exception as e:
            raise MyException(e, sys) from e

    def load_model(self, model_filepath: str, bucket_name: str) -> Any:
        """
        Load a pickled model from S3.

        Args:
            model_filename (str): The name of the model file
            model_dirpath (Optional[str]): The directory path within the bucket (can be None)
            bucket_name (str): The name of the S3 bucket

        Returns:
            Any: The unpickled model object

        Raises:
            MyException: If model loading fails
        """
        try:

            file_object = self.get_file_object(
                filename=model_filepath, bucket_name=bucket_name
            )

            if isinstance(file_object, list):
                if len(file_object) == 0:
                    raise FileNotFoundError(
                        f"S3 key prefix '{model_filepath}' not found in bucket '{bucket_name}'"
                    )
                file_object = file_object[0]

            model_content = self.read_object(object_name=file_object, decode=False)
            model = pickle.loads(model_content)

            return model

        except Exception as e:
            raise MyException(e, sys) from e

    def create_directory(self, dirname: str, bucket_name: str) -> None:
        """
        Create a directory in S3 bucket (by creating an empty object with trailing slash).

        Args:
            dirname (str): The directory name to create
            bucket_name (str): The name of the S3 bucket

        Raises:
            MyException: If directory creation fails
        """

        try:
            directory_key = dirname.rstrip("/") + "/"

            try:
                self.client.head_object(Bucket=bucket_name, Key=directory_key)
                return

            except ClientError as e:
                if e.response["Error"]["Code"] == "404":
                    self.client.put_object(Bucket=bucket_name, Key=directory_key)
                else:
                    raise MyException(e, sys) from e

        except ClientError as e:
            raise MyException(e, sys) from e

        except Exception as e:
            raise MyException(e, sys) from e

    def upload_file(
        self,
        from_filename: str,
        to_filename: str,
        bucket_name: str,
        remove: bool = True,
    ) -> None:
        """
        Upload a file to S3 bucket.

        Args:
            from_filename (str): Local file path to upload
            to_filename (str): S3 key name for the uploaded file
            bucket_name (str): The name of the S3 bucket
            remove (bool): Whether to remove the local file after upload. Defaults to True

        Raises:
            MyException: If file upload fails
        """
        try:
            if not os.path.exists(from_filename):
                raise FileNotFoundError(f"Local file '{from_filename}' does not exist")

            self.resource.meta.client.upload_file(
                from_filename, bucket_name, to_filename
            )

            if remove:
                os.remove(from_filename)

        except FileNotFoundError as e:
            raise MyException(e, sys) from e

        except ClientError as e:
            raise MyException(e, sys) from e

        except Exception as e:
            raise MyException(e, sys) from e

    def read_csv(self, filename: str, bucket_name: str) -> pd.DataFrame:
        """
        Read a CSV file from S3 and return as pandas DataFrame.

        Args:
            filename (str): The S3 key name of the CSV file
            bucket_name (str): The name of the S3 bucket

        Returns:
            pd.DataFrame: The CSV data as a pandas DataFrame

        Raises:
            MyException: If CSV reading fails
        """
        try:
            csv_object = self.get_file_object(
                filename=filename, bucket_name=bucket_name
            )

            if isinstance(csv_object, list):
                csv_object = csv_object[0]

            df = self.get_df_from_object(csv_object)
            return df

        except Exception as e:
            raise MyException(e, sys) from e

    def upload_df_as_csv(
        self,
        df: pd.DataFrame,
        local_filepath: str,
        bucket_filename: str,
        bucket_name: str,
    ) -> None:
        """
        Save DataFrame as CSV locally and upload to S3.

        Args:
            df (pd.DataFrame): The DataFrame to save and upload
            local_filepath (str): Local path to temporarily save the CSV file
            bucket_filename (str): S3 key name for the uploaded CSV file
            bucket_name (str): The name of the S3 bucket

        Raises:
            MyException: If DataFrame upload fails
        """
        try:
            if df.empty:
                logging.warning("Uploading empty DataFrame")

            save_df_as_csv(filepath=local_filepath, df=df, index=None, header=True)

            self.upload_file(
                from_filename=local_filepath,
                to_filename=bucket_filename,
                bucket_name=bucket_name,
            )

        except Exception as e:
            raise MyException(e, sys) from e

    def get_df_from_object(self, object_name: Object) -> pd.DataFrame:
        """
        Convert S3 object content to pandas DataFrame.

        Args:
            object_name (Object): The S3 object containing CSV data

        Returns:
            pd.DataFrame: The CSV data as a pandas DataFrame

        Raises:
            MyException: If DataFrame conversion fails
        """
        try:
            object_content = self.read_object(
                object_name=object_name, make_readable=True
            )
            df = read_csv_file(filepath=object_content)
            return df

        except Exception as e:
            raise MyException(e, sys) from e
