import os
import logging
import colorlog
from typing import Callable
from datetime import datetime
from logging.handlers import RotatingFileHandler


# Fallback if from_root is unavailable (returns current working directory; replace with your actual import)
def fallback_from_root() -> str:
    return os.getcwd()


# Uncomment the line below if your from_root works; otherwise use fallback
# from from_root import from_root  # type: Callable[[], str]
from_root: Callable[[], str] = fallback_from_root  # Use fallback for testing

# Initializing variables
LOGS_DIR: str = "logs"
LOG_FILE_FORMAT: str = f"{datetime.now().strftime('%d-%b-%y_%H:%M:%S')}.log"
maxBytes: int = 5 * 1024 * 1024  # 5 MB
backupCount: int = 4

# Full path to the logs directory, created if it doesn't exist
logs_dirpath: str = os.path.join(from_root(), LOGS_DIR)
os.makedirs(logs_dirpath, exist_ok=True)
log_filepath: str = os.path.join(logs_dirpath, LOG_FILE_FORMAT)


def config_logger() -> None:
    """
    Configures the root logger with a colored console handler and a rotating file handler.

    This function sets up logging to output to both the console (with color-coded levels)
    and a rotating file for persistent storage. Handlers are added only if none exist
    to prevent duplicates. The logger level is set to DEBUG for maximum verbosity.

    Handlers:
    - Console: Uses colorlog for level-specific colors, logs at DEBUG level.
    - File: Rotates files when they exceed maxBytes, keeps up to backupCount backups.

    Raises:
        Any exceptions from handler initialization (e.g., file permission issues).
    """
    # Get the root logger
    logger: logging.Logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Formatter for file and console logs
    file_format: logging.Formatter = logging.Formatter(
        "[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s"
    )
    console_format: colorlog.ColoredFormatter = colorlog.ColoredFormatter(
        "[ %(asctime)s ] %(name)s - %(log_color)s%(levelname)s%(reset)s - %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )

    if not logger.handlers:
        # Set up console handler
        console_handler: logging.StreamHandler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

        # Set up rotating file handler with explicit encoding
        file_handler: RotatingFileHandler = RotatingFileHandler(
            log_filepath, maxBytes=maxBytes, backupCount=backupCount, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)


# Initialize the logger configuration
config_logger()
