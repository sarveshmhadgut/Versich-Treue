import sys
from numpy import ndarray
from pandas import DataFrame
from typing import Dict, Any
from src.logger import logging
from src.exception import MyException
from sklearn.pipeline import Pipeline


class TargetMapping:
    """
    Class for mapping target labels between categorical and numerical representations.

    Maps 'yes' to 1 and 'no' to 0 for binary classification tasks.

    Attributes:
        yes (int): Numerical representation for positive class.
        no (int): Numerical representation for negative class.
    """

    def __init__(self) -> None:
        """
        Initialize TargetMapping with default yes=1, no=0 mapping.

        Raises:
            MyException: For unexpected initialization errors.
        """
        try:
            self.yes = 1
            self.no = 0

        except Exception as e:
            raise MyException(e, sys) from e

    def _asdict(self) -> Dict[str, int]:
        """
        Convert mapping to dictionary format.

        Returns:
            Dict[str, int]: Dictionary containing the mapping {label: value}.

        Raises:
            MyException: For unexpected errors during dictionary conversion.
        """
        try:
            mapping_dict = self.__dict__
            return mapping_dict

        except Exception as e:
            raise MyException(e, sys) from e

    def reverse_mapping(self) -> Dict[int, str]:
        """
        Get reverse mapping from numerical values to categorical labels.

        Returns:
            Dict[int, str]: Dictionary mapping numerical values to labels {value: label}.

        Raises:
            MyException: For unexpected errors during reverse mapping creation.
        """
        try:
            mapping_response = self._asdict()
            reverse_map = dict(zip(mapping_response.values(), mapping_response.keys()))
            return reverse_map

        except Exception as e:
            raise MyException(e, sys) from e


class Model:
    """
    Wrapper class that combines preprocessing pipeline and trained model for predictions.

    Attributes:
        preprocessor (Pipeline): Sklearn pipeline for data preprocessing.
        trained_model (object): Trained machine learning model.
    """

    def __init__(self, preprocessor: Pipeline, trained_model: object) -> None:
        """
        Initialize Model with preprocessor and trained model.

        Args:
            preprocessor (Pipeline): Sklearn preprocessing pipeline.
            trained_model (object): Trained machine learning model object.

        Raises:
            MyException: For initialization errors.
        """
        try:
            self.preprocessor = preprocessor
            self.trained_model = trained_model

        except Exception as e:
            raise MyException(e, sys) from e

    def predict(self, df: DataFrame) -> ndarray:
        """
        Make predictions on input data using the preprocessing pipeline and trained model.

        Args:
            df (pd.DataFrame): Input data for prediction.

        Returns:
            np.ndarray: Array of predictions.

        Raises:
            MyException: For prediction errors during preprocessing or model inference.
        """
        try:
            transformed_features = self.preprocessor.transform(df)
            predictions = self.trained_model.predict(transformed_features)
            return predictions

        except Exception as e:
            raise MyException(e, sys) from e

    def __repr__(self) -> str:
        """
        Return string representation of the model.

        Returns:
            str: String representation showing the trained model type.
        """
        try:
            return f"{type(self.trained_model).__name__}()"
        except Exception as e:
            logging.error(f"Error in __repr__: {e}")
            return "Model()"

    def __str__(self) -> str:
        """
        Return string representation of the model.

        Returns:
            str: String representation showing the trained model type.
        """
        try:
            return f"{type(self.trained_model).__name__}()"
        except Exception as e:
            logging.error(f"Error in __str__: {e}")
            return "Model()"
