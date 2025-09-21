import os
import sys
import certifi
import pymongo
from dotenv import load_dotenv
from src.logger import logging
from typing import Optional, Any
from src.exception import MyException
from src.constants import DATABASE_NAME, MONGODB_CONNECTION_URL

load_dotenv()
ca: Any = certifi.where()


class MongoDBClient:
    client: Optional[pymongo.MongoClient] = None

    def __init__(self, database_name: str = DATABASE_NAME) -> None:
        """
        Initialize a MongoDB client and establish connection to the specified database.

        Args:
            database_name (str): The name of the database to connect to. Defaults to DATABASE_NAME.

        Raises:
            MyException: If connection URL is not found or connection to MongoDB fails.
        """
        try:
            if MongoDBClient.client is None:

                logging.info("Connecting to MongoDB client...")
                connection_url: str = os.getenv(MONGODB_CONNECTION_URL)

                if connection_url is None:
                    raise MyException(
                        f"Environment variable {MONGODB_CONNECTION_URL} not set",
                        sys,
                    )

                MongoDBClient.client: pymongo.MongoClient = pymongo.MongoClient(
                    connection_url, tlsCAFile=ca
                )
                logging.info("MongoDB client initialized successfully.")

            logging.info(f"Connecting to database...")
            self.client: Any = MongoDBClient.client
            self.database: Any = self.client[database_name]

            logging.info(f"Connected to database: {database_name}")

        except Exception as e:
            raise MyException(e, sys)
