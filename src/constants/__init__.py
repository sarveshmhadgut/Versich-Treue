# mongo setup
DATABASE_NAME: str = "Versich-Treue"
COLLECTION_NAME: str = "Versich-Treue-Data"
MONGODB_CONNECTION_URL: str = "MONGODB_CONNECTION_URL"

# data ingestion
PIPELINE_NAME: str = ""
ARTIFACT_PATHNAME: str = "artifacts"
DATA_INGESTION_DIRNAME: str = "data_ingestion"
FEATURE_STORE_DIRNAME: str = "feature_store"
DATA_FILENAME: str = "data.csv"
DATA_INGESTION_INGESTED_DIRNAME: str = "ingested"
TRAIN_DATA_FILENAME = "train.csv"
TEST_DATA_FILENAME = "test.csv"
TEST_SIZE: int = 0.2
