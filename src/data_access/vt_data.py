import sys
from halo import Halo
from numpy import nan
from pandas import DataFrame
from typing import Optional, Any
from src.exception import MyException
from src.constants import DATABASE_NAME
from src.configuration.mongo_db_connection import MongoDBClient


class VTData:
    def __init__(self) -> None:
        """
        Initializes the VTData instance by creating a MongoDB client connection.

        Raises:
            MyException: If the MongoDB client initialization fails.
        """
        try:
            self.client: MongoDBClient = MongoDBClient(database_name=DATABASE_NAME)

        except Exception as e:
            raise MyException(e, sys) from e

    def export_collection_as_dataframe(
        self, collection_name: str, database_name: Optional[str] = None
    ) -> DataFrame:
        """
        Export data from a MongoDB collection to a pandas DataFrame.

        Args:
            collection_name (str): The name of the MongoDB collection to export.
            database_name (Optional[str]): The database name. If None, uses the default database from client.

        Returns:
            pd.DataFrame: DataFrame containing data from the specified collection.

        Raises:
            MyException: If data retrieval or conversion fails.
        """
        try:

            if database_name is None:
                collection: Any = self.client.database[collection_name]

            else:
                collection: Any = self.client.database.client[database_name][
                    collection_name
                ]

            with Halo(text="Fetching records...", spinner="dots"):
                data: list = list(collection.find())

            df: DataFrame = DataFrame(data)

            if "_id" in df.columns:
                df.drop(columns=["_id"], inplace=True, axis=1)

            df.replace({"na": nan}, inplace=True)
            return df

        except Exception as e:
            raise MyException(e, sys) from e
