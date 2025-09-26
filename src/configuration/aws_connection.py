import os
import sys
import boto3
from dotenv import load_dotenv
from typing import Optional, ClassVar
from src.exception import MyException
from mypy_boto3_s3.client import S3Client
from mypy_boto3_s3.service_resource import S3ServiceResource
from botocore.exceptions import NoCredentialsError, ClientError, BotoCoreError
from src.constants import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION


class S3:
    """
    A singleton S3 connection manager that provides cached client and resource instances.

    This class implements a singleton pattern to ensure that S3 client and resource
    instances are created only once and reused across the application, improving
    performance by avoiding repeated authentication overhead.

    Attributes:
        client (ClassVar[Optional[S3Client]]): Shared S3 client instance
        resource (ClassVar[Optional[S3ServiceResource]]): Shared S3 resource instance
    """

    client: ClassVar[Optional[S3Client]] = None
    resource: ClassVar[Optional[S3ServiceResource]] = None

    def __init__(self, region_name: str = AWS_REGION) -> None:
        """
        Initialize S3 connection with cached client and resource instances.

        Creates S3 client and resource instances using environment variables for
        AWS credentials. Implements singleton pattern to reuse existing instances
        if they have already been created.

        Args:
            region_name (str): AWS region name for the S3 connection. Defaults to AWS_REGION constant.

        Raises:
            MyException: If AWS credentials are missing or invalid
            MyException: If S3 client/resource creation fails

        Note:
            Environment variables AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
            must be set before creating an instance.
        """
        try:
            load_dotenv()

            if not S3.client or not S3.resource:
                access_key = os.getenv(AWS_ACCESS_KEY_ID)
                secret_key = os.getenv(AWS_SECRET_ACCESS_KEY)
                region_name = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

                if not access_key:
                    raise MyException(
                        f"AWS Access Key ID not found '{AWS_ACCESS_KEY_ID}'",
                        sys,
                    )

                if not secret_key:
                    raise MyException(
                        f"AWS Secret Access Key not found '{AWS_SECRET_ACCESS_KEY}'",
                        sys,
                    )

                S3.client = boto3.client(
                    "s3",
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                    region_name=region_name,
                )

                S3.resource = boto3.resource(
                    "s3",
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                    region_name=region_name,
                )

            self.client: S3Client = S3.client
            self.resource: S3ServiceResource = S3.resource

        except MyException:
            raise

        except NoCredentialsError as e:
            raise MyException(e, sys) from e

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            raise MyException(f"AWS S3 client error ({error_code}): {e}", sys) from e

        except BotoCoreError as e:
            raise MyException(e, sys) from e

        except Exception as e:
            raise MyException(e, sys) from e
