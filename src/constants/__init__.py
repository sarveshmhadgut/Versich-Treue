import os

PIPELINE_NAME: str = ""
ARTIFACT_PATHNAME: str = "artifacts"
SCHEMA_FILEPATH = os.path.join("config", "schema.yaml")

# mongodb setup
DATABASE_NAME: str = "Versich-Treue"
COLLECTION_NAME: str = "Versich-Treue-Data"
MONGODB_CONNECTION_URL: str = "MONGODB_CONNECTION_URL"

# data ingestion
DATA_INGESTION_DIRNAME: str = "data_ingestion"
FEATURE_STORE_DIRNAME: str = "feature_store"
DATA_FILENAME: str = "data.csv"
DATA_INGESTION_INGESTED_DIRNAME: str = "ingested"
TRAIN_DATA_FILENAME = "train.csv"
TEST_DATA_FILENAME = "test.csv"
TEST_SIZE: int = 0.2

# data validation
DATA_VALIDATION_DIRNAME = "data_validation"
DATA_VALIDATION_REPORT_DIRNAME = "reports"
DATA_VALIDATION_REPORT_FILENAME = "report.yaml"
