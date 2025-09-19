import os
import sys
from src.logger import logging
from src.exception import MyException

logging.info("Starting rotation test in test.py...")
for i in range(100000):  # This should generate ~5-10MB of logs; adjust range as needed
    logging.debug(
        f"Test log message {i} from test.py - This should force file rotation."
    )

logging.info("Test complete.")


try:
    # Intentionally cause a ZeroDivisionError for testing
    result = 1 / 0
except Exception as e:
    logging.info(e)
    raise MyException(e, sys) from e
