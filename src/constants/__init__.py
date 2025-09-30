import os

PIPELINE_NAME: str = ""
ARTIFACT_PATHNAME: str = "artifacts"
SCHEMA_FILEPATH = os.path.join("config", "schema.yaml")
MODEL_PARAMS_FILEPATH = os.path.join("config", "model.yaml")
TRAIN_DATA_FILENAME = "train.csv"
TEST_DATA_FILENAME = "test.csv"
REPORT_DIRNAME = "reports"
REPORT_FILENAME = "report.yaml"

# mongodb setup
DATABASE_NAME: str = "Versich-Treue"
COLLECTION_NAME: str = "Versich-Treue-Data"
MONGODB_CONNECTION_URL: str = "MONGODB_CONNECTION_URL"

# data ingestion
DATA_INGESTION_DIRNAME: str = "data_ingestion"
FEATURE_STORE_DIRNAME: str = "feature_store"
DATA_INGESTION_INGESTED_DIRNAME: str = "ingested"
DATA_FILENAME: str = "data.csv"
TEST_SIZE: int = 0.2

# data validation
DATA_VALIDATION_DIRNAME: str = "data_validation"

# data transformation
DATA_TRANSFORMATION_DIRNAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIRPATH: str = "transformed"
DATA_TRANSFORMATION_OBJECT_DIRNAME: str = "objects"
DATA_TRANSFORMATION_OBJECT_FILENAME: str = "data_transformation_object"

# model training
MODEL_TRAINING_DIRNAME: str = "model_training"
TRAINED_MODEL_DIRNAME: str = "trained_model"
MODEL_FILENAME: str = "model.pkl"

# aws setup
AWS_ACCESS_KEY_ID: str = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY: str = "AWS_SECRET_ACCESS_KEY"
AWS_REGION: str = "AWS_DEFAULT_REGION"

# model evaluation
MODEL_EVALUATION_DIRNAME = "model_evaluation"
MODEL_EVALUATION_THRESHOLD: str = 0.05
MODEL_BUCKET_NAME: str = "versich-treue"

# app
APP_HOST = "localhost"
APP_PORT = 8080
