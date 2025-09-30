# import os
# import sys
# from src.logger import logging


# logging.info("Starting rotation test in test.py...")
# for i in range(100000):  # This should generate ~5-10MB of logs; adjust range as needed
#     logging.debug(
#         f"Test log message {i} from test.py - This should force file rotation."
#     )

# logging.info("Test complete.")

# from src.exception import MyException

# try:
#     # Intentionally cause a ZeroDivisionError for testing
#     result = 1 / 0
# except Exception as e:
#     logging.info(e)
#     raise MyException(e, sys) from e
"""
Validation script to check if transformation artifacts are generated correctly.
"""

# import numpy as np
# import os
# import sys
# from src.utils.main_utils import load_object, load_numpy_array


# def validate_transformation_artifacts():
#     """Validate the transformation artifacts."""

#     # Find the most recent artifacts folder
#     artifacts_base = "artifacts"
#     if not os.path.exists(artifacts_base):
#         print(f"Artifacts directory '{artifacts_base}' not found!")
#         return False

#     # Get all timestamp directories
#     timestamp_dirs = [d for d in os.listdir(artifacts_base)
#                      if os.path.isdir(os.path.join(artifacts_base, d))]

#     if not timestamp_dirs:
#         print(f"No timestamp directories found in '{artifacts_base}'!")
#         return False

#     # Sort by timestamp (newest first) and take the most recent
#     timestamp_dirs.sort(reverse=True)
#     latest_timestamp = timestamp_dirs[0]
#     artifacts_path = os.path.join(artifacts_base, latest_timestamp, "data_transformation", "transformed_data")

#     print(f"Using artifacts from: {artifacts_path}")

#     print("=" * 60)
#     print("TRANSFORMATION ARTIFACTS VALIDATION")
#     print("=" * 60)

#     train_file = os.path.join(artifacts_path, "train.npy")
#     test_file = os.path.join(artifacts_path, "test.npy")
#     transformer_file = os.path.join(artifacts_path, "data_transformation_object")

#     files_to_check = [
#         ("Training data", train_file),
#         ("Test data", test_file),
#         ("Transformer object", transformer_file),
#     ]

#     all_files_exist = True

#     for name, filepath in files_to_check:
#         if os.path.exists(filepath):
#             print(f"{name}: {filepath} - EXISTS")
#         else:
#             print(f"{name}: {filepath} - MISSING")
#             all_files_exist = False

#     if not all_files_exist:
#         print("\nSome artifacts are missing!")
#         return False

#     print("\n" + "-" * 60)
#     print("LOADING AND VALIDATING ARTIFACTS...")
#     print("-" * 60)

#     try:
#         train_array = load_numpy_array(train_file)
#         print(f"Training data loaded successfully")
#         print(f"  - Shape: {train_array.shape}")
#         print(f"  - Data type: {train_array.dtype}")
#         print(f"  - Sample values (first 3 rows, first 5 columns):")
#         print(f"    {train_array[:3, :5]}")

#         test_array = load_numpy_array(test_file)
#         print(f"\nTest data loaded successfully")
#         print(f"  - Shape: {test_array.shape}")
#         print(f"  - Data type: {test_array.dtype}")
#         print(f"  - Sample values (first 3 rows, first 5 columns):")
#         print(f"    {test_array[:3, :5]}")

#         transformer = load_object(transformer_file)
#         print(f"\nTransformer object loaded successfully")
#         print(f"  - Type: {type(transformer)}")
#         print(f"  - Pipeline steps: {transformer.steps}")

#         print(f"\n" + "-" * 60)
#         print("DATA INTEGRITY CHECKS...")
#         print("-" * 60)

#         train_nan_count = np.isnan(train_array).sum()
#         test_nan_count = np.isnan(test_array).sum()

#         print(f"Training data NaN values: {train_nan_count}")
#         print(f"Test data NaN values: {test_nan_count}")

#         train_inf_count = np.isinf(train_array).sum()
#         test_inf_count = np.isinf(test_array).sum()

#         print(f"Training data infinite values: {train_inf_count}")
#         print(f"Test data infinite values: {test_inf_count}")

#         expected_features = 13
#         if train_array.shape[1] >= expected_features:
#             print(
#                 f"Training data has expected number of columns ({train_array.shape[1]})"
#             )
#         else:
#             print(
#                 f"Training data has unexpected number of columns ({train_array.shape[1]}, expected >= {expected_features})"
#             )

#         if test_array.shape[1] >= expected_features:
#             print(f"Test data has expected number of columns ({test_array.shape[1]})")
#         else:
#             print(
#                 f"Test data has unexpected number of columns ({test_array.shape[1]}, expected >= {expected_features})"
#             )

#         if train_array.shape[1] >= 14:
#             target_column = train_array[:, -1]
#             unique_values, counts = np.unique(target_column, return_counts=True)
#             print(f"\nTarget variable distribution in training data:")
#             for val, count in zip(unique_values, counts):
#                 percentage = (count / len(target_column)) * 100
#                 print(f"  - Class {val}: {count} samples ({percentage:.2f}%)")

#         print(f"\n" + "=" * 60)
#         print("ALL TRANSFORMATION ARTIFACTS ARE VALID!")
#         print("=" * 60)

#         return True

#     except Exception as e:
#         print(f"\nError loading artifacts: {str(e)}")
#         return False


# if __name__ == "__main__":
#     success = validate_transformation_artifacts()
#     sys.exit(0 if success else 1)

# from src.pipeline.training_pipeline import TrainPipeline

# pipeline = TrainPipeline()
# pipeline.run_pipeline()

# test II
