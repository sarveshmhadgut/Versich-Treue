import sys
import numpy as np
from halo import Halo
from pandas import DataFrame
from src.logger import logging
from src.exception import MyException
from src.entity.estimator import Model
from sklearn.ensemble import RandomForestClassifier
from src.entity.config_entity import ModelTrainingConfig
from src.utils.main_utils import (
    load_numpy_array,
    load_object,
    save_object,
    save_as_json,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    log_loss,
    f1_score,
    roc_auc_score,
)
from src.entity.artifact_entity import (
    ModelTrainingArtifacts,
    DataTransformationArtifacts,
    ClassificationMetricsArtifacts,
)


class ModelTraining:
    def __init__(
        self,
        data_transformation_artifacts: DataTransformationArtifacts,
        model_training_config: ModelTrainingConfig,
    ) -> None:
        """
        Initialize ModelTraining class with required artifacts and configuration.

        Args:
            data_transformation_artifacts (DataTransformationArtifacts): Artifacts from data transformation stage containing file paths.
            model_training_config (ModelTrainingConfig): Configuration parameters for model training.

        Raises:
            None
        """
        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_training_config = model_training_config

    def get_trained_model(self, train: DataFrame) -> RandomForestClassifier:
        """
        Train a RandomForestClassifier using training data.

        Args:
            train (pd.DataFrame): Numpy array with features and labels for training, last column is label.

        Returns:
            RandomForestClassifier: Trained Random Forest classifier.

        Raises:
            MyException: For unexpected errors in training.
        """
        try:
            X_train, y_train = train[:, :-1], train[:, -1]
            classifier = RandomForestClassifier(
                **self.model_training_config.training_model_params
            )

            with Halo(text="Training random forest classifier...", spinner="dots"):
                classifier.fit(X_train, y_train)
            return classifier

        except Exception as e:
            raise MyException(e, sys) from e

    def get_classification_report(
        self, classifier: RandomForestClassifier, test: DataFrame
    ) -> dict:
        """
        Generate classification metrics on test dataset.

        Args:
            classifier (RandomForestClassifier): Trained classifier.
            test (pd.DataFrame): Numpy array with features and labels for testing, last column is label.

        Returns:
            dict: Dictionary with accuracy, precision, recall, log_loss, f1_score, roc_auc.

        Raises:
            MyException: For unexpected errors during metric calculation.
        """
        try:
            X_test, y_test = test[:, :-1], test[:, -1]

            y_hat = classifier.predict(X=X_test)
            y_hat_proba = classifier.predict_proba(X=X_test)

            metrics = {
                "accuracy": round(accuracy_score(y_true=y_test, y_pred=y_hat), 5),
                "precision": round(precision_score(y_true=y_test, y_pred=y_hat), 5),
                "recall": round(recall_score(y_true=y_test, y_pred=y_hat), 5),
                "log_loss_": round(log_loss(y_true=y_test, y_pred=y_hat_proba), 5),
                "f1_score_": round(f1_score(y_true=y_test, y_pred=y_hat), 5),
                "roc_auc": round(
                    roc_auc_score(y_true=y_test, y_score=y_hat_proba[:, 1]), 5
                ),
            }

            return metrics

        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_training(self) -> ModelTrainingArtifacts:
        """
        Run the complete model training pipeline:
        - Load transformed train and test datasets
        - Train model
        - Generate classification metrics
        - Load preprocessing object
        - Validate metrics against threshold
        - Save trained model and metrics report
        - Create and return training artifacts object

        Returns:
            ModelTrainingArtifacts: Artifacts generated as a result of model training process

        Raises:
            MyException: If any step in training pipeline fails or accuracy threshold not met
        """
        try:
            train = load_numpy_array(
                filepath=self.data_transformation_artifacts.data_transformation_train_array_filepath
            )

            test = load_numpy_array(
                filepath=self.data_transformation_artifacts.data_transformation_test_array_filepath
            )
            logging.info("Training & testing data loaded")

            classifier = self.get_trained_model(train=train)
            logging.info("Random forest classifier model trained")

            metrics = self.get_classification_report(classifier=classifier, test=test)
            logging.info("Classification report drafted")

            preprocessor = load_object(
                filepath=self.data_transformation_artifacts.data_transformation_object_filepath
            )
            logging.info("Preprocessor object fetched")

            if metrics["accuracy"] < self.model_training_config.threshold_accuracy:
                error_msg = f"Model accuracy {metrics['accuracy']:.4f} below threshold {self.model_training_config.threshold_accuracy}"
                raise MyException(error_msg, sys)

            pipeline = Model(preprocessor=preprocessor, trained_model=classifier)
            logging.info("Trained model fetched")

            save_object(
                obj=pipeline,
                filepath=self.model_training_config.trained_model_filepath,
            )
            logging.info("Trained model saved")

            save_as_json(
                data=metrics,
                filepath=self.model_training_config.report_filepath,
                indent=4,
            )
            logging.info("Classification report saved")

            classification_artifacts = ClassificationMetricsArtifacts(
                accuracy=metrics["accuracy"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                log_loss_=metrics["log_loss_"],
                f1_score_=metrics["f1_score_"],
                roc_auc=metrics["roc_auc"],
            )

            model_training_artifacts = ModelTrainingArtifacts(
                trained_model_filepath=self.model_training_config.trained_model_filepath,
                report_filepath=self.model_training_config.report_filepath,
                classification_metrics_artifacts=classification_artifacts,
            )
            return model_training_artifacts

        except Exception as e:
            raise MyException(e, sys) from e
