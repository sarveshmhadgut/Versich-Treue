import sys
import numpy as np
import pandas as pd
from halo import Halo
from src.logger import logging
from src.exception import MyException
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from src.constants import SCHEMA_FILEPATH
from sklearn.compose import ColumnTransformer
from src.entity.config_entity import DataTransformationConfig
from src.utils.main_utils import (
    save_object,
    read_csv_file,
    read_yaml_file,
    save_numpy_array,
)
from src.entity.artifact_entity import (
    DataIngestionArtifacts,
    DataValidationArtifacts,
    DataTransformationArtifacts,
)
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    LabelEncoder,
    OneHotEncoder,
)


class DataTransformation:
    def __init__(
        self,
        data_ingestion_artifacts: DataIngestionArtifacts,
        data_validation_artifacts: DataValidationArtifacts,
        data_transformation_config: DataTransformationConfig,
    ) -> None:
        """
        Initialize the DataTransformation class.

        Args:
            data_ingestion_artifacts (DataIngestionArtifacts): Artifacts from data ingestion.
            data_validation_artifacts (DataValidationArtifacts): Artifacts from data validation.
            data_transformation_config (DataTransformationConfig): Configuration for data transformation.
        """
        try:
            self.data_ingestion_artifacts = data_ingestion_artifacts
            self.data_validation_artifacts = data_validation_artifacts
            self.data_transformation_config = data_transformation_config
            self.schema_config = read_yaml_file(SCHEMA_FILEPATH)

        except Exception as e:
            raise MyException(e, sys) from e

    def _drop_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop unwanted features from the dataframe as per schema.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe after dropping specified features.
        """
        try:
            drop_features = self.schema_config.get("drop_features", [])

            for col in drop_features:
                if col in df.columns:
                    df.drop(col, axis=1, inplace=True)

            return df

        except Exception as e:
            raise MyException(e, sys) from e

    def _label_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply label encoding to specified features.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with label encoded features.
        """
        try:
            label_encoding_features = self.schema_config.get(
                "label_encoding_features", []
            )
            for col in label_encoding_features:
                if col in df.columns:
                    le = LabelEncoder()
                    df[col] = df[col].fillna("Unknown").astype(str)
                    df[col] = le.fit_transform(df[col])

            return df

        except Exception as e:
            raise MyException(e, sys) from e

    def _onehot_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply one-hot encoding to specified features.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with one-hot encoded features.
        """
        try:
            onehot_encoding_features = self.schema_config.get(
                "onehot_encoding_features", []
            )
            oh_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

            for feature in onehot_encoding_features:
                if feature in df.columns:
                    df[feature] = df[feature].astype(str)
                    encoded = oh_encoder.fit_transform(df[[feature]])
                    col_names = oh_encoder.get_feature_names_out([feature])
                    encoded_df = pd.DataFrame(
                        encoded, columns=col_names, index=df.index
                    ).astype(int)
                    df = pd.concat([df.drop(columns=[feature]), encoded_df], axis=1)

            return df

        except Exception as e:
            raise MyException(e, sys) from e

    def _rename_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename dataframe features according to schema config.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with renamed columns.
        """
        try:
            rename_features = {
                col[0]: col[1]
                for col in [
                    c.split("$") for c in self.schema_config.get("rename_features", [])
                ]
            }

            df.rename(columns=rename_features, inplace=True)
            return df

        except Exception as e:
            raise MyException(e, sys) from e

    def _float_to_int(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert all float columns in dataframe to integer.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with floats converted to int.
        """
        try:
            float_cols = df.select_dtypes(include=["float"]).columns
            for col in float_cols:
                df[col] = df[col].astype(int)

            return df

        except Exception as e:
            raise MyException(e, sys) from e

    def get_data_transformer(self) -> Pipeline:
        """
        Create and return the data transformation pipeline including scaling.

        Returns:
            Pipeline: Sklearn pipeline with data transformation steps.
        """
        try:
            normalizer = MinMaxScaler()
            standardizer = StandardScaler()

            normalization_features = self.schema_config.get(
                "normalization_features", []
            )
            standardization_features = self.schema_config.get(
                "standardization_features", []
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("Normalizer", normalizer, normalization_features),
                    ("Standardizer", standardizer, standardization_features),
                ],
                remainder="passthrough",
            )

            pipeline = Pipeline(steps=[("Preprocessing", preprocessor)])
            return pipeline

        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifacts:
        """
        Perform complete data transformation pipeline including encoding, scaling, and sampling.

        Returns:
            DataTransformationArtifacts: Artifacts generated after transformation.
        """
        try:
            logging.info("Starting data transformation process...")

            train_df = read_csv_file(
                filepath=self.data_ingestion_artifacts.train_filepath
            )
            logging.info("Training data read.")

            test_df = read_csv_file(
                filepath=self.data_ingestion_artifacts.test_filepath
            )
            logging.info("Testing data read.")

            target_features = self.schema_config.get("target_features", [])
            logging.info(f"Target features identified: {target_features}")

            train_target_df = train_df.loc[:, target_features]
            train_input_df = train_df.drop(target_features, axis=1)

            test_target_df = test_df.loc[:, target_features]
            test_input_df = test_df.drop(target_features, axis=1)

            train_input_df = self._drop_features(train_input_df)
            logging.info("Features dropped from training data.")

            train_input_df = self._label_encoding(train_input_df)
            logging.info("Label encoding applied on training data.")

            train_input_df = self._onehot_encoding(train_input_df)
            logging.info("One-hot encoding applied on training data.")

            train_input_df = self._rename_features(train_input_df)
            logging.info("Features renamed from training data.")

            test_input_df = self._drop_features(test_input_df)
            logging.info("Features dropped from testing data.")

            test_input_df = self._label_encoding(test_input_df)
            logging.info("Label encoding applied on testing data.")

            test_input_df = self._onehot_encoding(test_input_df)
            logging.info("One-hot encoding applied on testing data.")

            test_input_df = self._rename_features(test_input_df)
            logging.info("Features renamed from testing data.")

            preprocessor = self.get_data_transformer()
            logging.info("Preprocessing pipeline fetched.")

            train_input_arr = preprocessor.fit_transform(train_input_df)
            test_input_arr = preprocessor.transform(test_input_df)
            logging.info("Training & testing data transformed")

            smoteenn = SMOTEENN(sampling_strategy="minority")

            with Halo(text="Over-sampling train data...", spinner="dots") as spinner:
                final_train_inputs, final_train_targets = smoteenn.fit_resample(
                    X=train_input_arr, y=train_target_df.values.ravel()
                )
                logging.info("Training data over-sampled")

            with Halo(text="Over-sampling test data...", spinner="dots") as spinner:
                final_test_inputs, final_test_targets = smoteenn.fit_resample(
                    X=test_input_arr, y=test_target_df.values.ravel()
                )
                logging.info("Testing data over-sampled")

            train_array = np.c_[final_train_inputs, final_train_targets]
            test_array = np.c_[final_test_inputs, final_test_targets]

            save_object(
                preprocessor,
                self.data_transformation_config.data_transformation_object_filepath,
            )
            logging.info("Preprocessor object saved")

            save_numpy_array(
                np_array=train_array,
                filepath=self.data_transformation_config.data_transformation_train_array_filepath,
            )
            logging.info("Training array saved")

            save_numpy_array(
                np_array=test_array,
                filepath=self.data_transformation_config.data_transformation_test_array_filepath,
            )
            logging.info("Testing array saved")

            logging.info("Data transformation process completed.")
            return DataTransformationArtifacts(
                data_transformation_object_filepath=self.data_transformation_config.data_transformation_object_filepath,
                data_transformation_test_filepath=self.data_transformation_config.data_transformation_test_array_filepath,
                data_transformation_train_filepath=self.data_transformation_config.data_transformation_train_array_filepath,
            )

        except Exception as e:
            raise MyException(e, sys) from e
