import sys
import pandas as pd
from pandas import DataFrame
from src.logger import logging
from src.exception import MyException
from typing import Optional, Dict, Any, List
from src.entity.s3_estimator import S3Estimator
from src.entity.config_entity import OwnerClassifierConfig


class VehicleOwner:
    """
    Represents a vehicle owner with their insurance profile data.

    This class encapsulates vehicle owner information and provides methods
    to encode categorical features and convert data to various formats
    suitable for machine learning model predictions.

    Attributes:
        age (Optional[int]): Age of the vehicle owner in years.
        gender (Optional[str]): Gender of the vehicle owner ('Male' or 'Female').
        vintage (Optional[int]): Number of days since the customer first became associated with the company.
        region_code (Optional[float]): Unique code for the region of the customer.
        annual_premium (Optional[float]): Amount the customer needs to pay as premium in the year.
        vehicle_damage (Optional[str]): Whether customer had their vehicle damaged in the past ('Yes' or 'No').
        driving_license (Optional[int]): Whether the customer has a valid driving license (0 or 1).
        previously_insured (Optional[int]): Whether the customer already has vehicle insurance (0 or 1).
        policy_sales_channel (Optional[float]): Anonymized code for the channel of customer outreach.
        vehicle_age_1_2_year (Optional[int]): Whether vehicle age is between 1-2 years (0 or 1).
        vehicle_age_lt_1_year (Optional[int]): Whether vehicle age is less than 1 year (0 or 1).
        vehicle_age_gt_2_years (Optional[int]): Whether vehicle age is greater than 2 years (0 or 1).
    """

    def __init__(
        self,
        age: Optional[int],
        gender: Optional[str],
        vintage: Optional[int],
        region_code: Optional[float],
        annual_premium: Optional[float],
        vehicle_damage: Optional[str],
        driving_license: Optional[int],
        previously_insured: Optional[int],
        policy_sales_channel: Optional[float],
        vehicle_age_1_2_year: Optional[int],
        vehicle_age_lt_1_year: Optional[int],
        vehicle_age_gt_2_years: Optional[int],
    ) -> None:
        """
        Initialize VehicleOwner with insurance profile data.

        Args:
            age: Age of the vehicle owner in years.
            gender: Gender of the vehicle owner ('Male' or 'Female').
            vintage: Number of days since customer first association with company.
            region_code: Unique code for customer's region.
            annual_premium: Annual premium amount to be paid.
            vehicle_damage: Past vehicle damage status ('Yes' or 'No').
            driving_license: Valid driving license status (0 or 1).
            previously_insured: Previous insurance status (0 or 1).
            policy_sales_channel: Anonymized customer outreach channel code.
            vehicle_age_1_2_year: Vehicle age 1-2 years indicator (0 or 1).
            vehicle_age_lt_1_year: Vehicle age less than 1 year indicator (0 or 1).
            vehicle_age_gt_2_years: Vehicle age greater than 2 years indicator (0 or 1).
        """

        self.age = age
        self.gender = gender
        self.vintage = vintage
        self.region_code = region_code
        self.annual_premium = annual_premium
        self.vehicle_damage = vehicle_damage
        self.driving_license = driving_license
        self.previously_insured = previously_insured
        self.policy_sales_channel = policy_sales_channel
        self.vehicle_age_1_2_year = vehicle_age_1_2_year
        self.vehicle_age_lt_1_year = vehicle_age_lt_1_year
        self.vehicle_age_gt_2_years = vehicle_age_gt_2_years

    def _encode_categorical_features(self) -> Dict[str, Any]:
        """
        Encode categorical features into numerical values for model processing.

        This method converts categorical variables (gender, vehicle_damage) into
        numerical representations suitable for machine learning algorithms.

        Returns:
            Dict[str, Any]: Dictionary containing encoded categorical features.
                - gender_encoded: 1 for 'Female', 0 for 'Male' or other values
                - vehicle_damage_encoded: 1 for 'Yes', 0 for 'No' or other values

        Raises:
            MyException: If encoding process fails due to invalid data types or values.
        """
        try:

            gender_encoded = 1 if self.gender == "Female" else 0
            vehicle_damage_encoded = 1 if self.vehicle_damage == "Yes" else 0

            encoded_values = {
                "gender_encoded": gender_encoded,
                "vehicle_damage_encoded": vehicle_damage_encoded,
            }

            return encoded_values

        except Exception as e:
            raise MyException(e, sys) from e

    def vehicle_owner_as_dict(self) -> Dict[str, Any]:
        """
        Convert vehicle owner data to dictionary format with encoded features.

        This method creates a dictionary representation of the vehicle owner
        data with categorical features properly encoded and column names
        matching the expected model input format.

        Returns:
            Dict[str, Any]: Dictionary containing all vehicle owner features
                with proper column names and encoded categorical variables.

        Raises:
            MyException: If data conversion or encoding fails.
        """
        try:

            encoded_values = self._encode_categorical_features()

            owner_dict = {
                "Age": self.age,
                "Gender": encoded_values["gender_encoded"],
                "Vintage": self.vintage,
                "Region_Code": self.region_code,
                "Annual_Premium": self.annual_premium,
                "Vehicle_Damage": encoded_values["vehicle_damage_encoded"],
                "Driving_License": self.driving_license,
                "Previously_Insured": self.previously_insured,
                "Policy_Sales_Channel": self.policy_sales_channel,
                "Vehicle_Age_1_2_Year": self.vehicle_age_1_2_year,
                "Vehicle_Age_lt_1_Year": self.vehicle_age_lt_1_year,
                "Vehicle_Age_gt_2_Years": self.vehicle_age_gt_2_years,
            }

            return owner_dict

        except Exception as e:
            raise MyException(e, sys) from e

    def vehicle_owner_as_df(self) -> DataFrame:
        """
        Convert vehicle owner data to pandas DataFrame format for model input.

        This method creates a properly formatted pandas DataFrame with correct
        column ordering, data types, and structure expected by the ML model.
        All categorical features are encoded and numeric columns are converted
        to appropriate data types.

        Returns:
            DataFrame: Single-row pandas DataFrame containing vehicle owner data
                with columns in expected order and proper data types.

        Raises:
            MyException: If DataFrame creation, column reordering, or type
                conversion fails.
        """
        try:

            vehicle_dict = self.vehicle_owner_as_dict()

            vehicle_df_data = {key: [value] for key, value in vehicle_dict.items()}
            df = DataFrame(vehicle_df_data)

            expected_column_order = [
                "Age",
                "Gender",
                "Vintage",
                "Region_Code",
                "Annual_Premium",
                "Vehicle_Damage",
                "Driving_License",
                "Previously_Insured",
                "Policy_Sales_Channel",
                "Vehicle_Age_1_2_Year",
                "Vehicle_Age_lt_1_Year",
                "Vehicle_Age_gt_2_Years",
            ]

            df = df.reindex(columns=expected_column_order)

            numeric_columns = expected_column_order

            for col in numeric_columns:

                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    logging.debug(f"Converted column '{col} to numeric type")

            return df

        except Exception as e:
            raise MyException(e, sys) from e


class OwnerClassifier:
    """
    Machine learning classifier for predicting vehicle insurance interest.

    This class handles the prediction pipeline for determining whether a vehicle
    owner is likely to be interested in vehicle insurance based on their profile.
    It loads a trained model from S3 storage and provides prediction functionality.

    Attributes:
        prediction_pipeline_config (OwnerClassifierConfig): Configuration object
            containing model file paths and S3 bucket information.
    """

    def __init__(
        self,
        prediction_pipeline_config: OwnerClassifierConfig = OwnerClassifierConfig(),
    ) -> None:
        """
        Initialize the OwnerClassifier with configuration settings.

        Args:
            prediction_pipeline_config: Configuration object containing model
                file paths and S3 bucket settings. Defaults to OwnerClassifierConfig().

        Raises:
            MyException: If configuration initialization fails.
        """
        try:
            self.prediction_pipeline_config = prediction_pipeline_config

        except Exception as e:
            raise MyException(e, sys) from e

    def predict(self, df: DataFrame) -> List[int]:
        """
        Generate predictions for vehicle insurance interest based on input data.

        This method validates the input DataFrame, loads the trained model from
        S3 storage, and generates predictions for vehicle insurance interest.

        Args:
            df: Input pandas DataFrame containing vehicle owner features.
                Must have all numeric columns and proper feature structure.

        Returns:
            List[int]: List of prediction results where each value represents
                the predicted insurance interest (typically 0 or 1).

        Raises:
            MyException: If input validation fails, model loading fails, or
                prediction generation encounters errors.
            ValueError: If DataFrame contains non-numeric data that cannot
                be processed by the ML model.
        """
        try:
            logging.info("Executing prediction pipeline...")

            for col in df.columns:
                if df[col].dtype == "object":
                    error_msg = (
                        f"Column '{col}' contains non-numeric data "
                        f"that cannot be processed by ML model"
                    )
                    raise ValueError(error_msg)

            logging.debug("All DataFrame columns validated")

            model = S3Estimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_filepath=self.prediction_pipeline_config.model_filepath,
            )
            logging.info("Model fetched from S3")

            result = model.tranform_predict(df)

            logging.debug("Prediction ready")
            return result

        except ValueError as ve:
            raise ve

        except Exception as e:
            raise MyException(e, sys) from e
