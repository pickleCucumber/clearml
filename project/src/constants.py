import os

from utils.log import setup_logger

logger = setup_logger(__name__)


def create_directory_if_not_exists(directory_path):
    """Creates a directory if it doesn't already exist and logs the action."""
    try:
        os.makedirs(directory_path, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create directory {directory_path}: {e}")


RANDOM_STATE = 42

# Base and root directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(BASE_DIR, ".."))

# Paths to specific directories
SQL_QUERY_PATH = os.path.join(BASE_DIR, "scripts")
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PREPARED_DATA_DIR = os.path.join(DATA_DIR, "prepared")
MODEL_DIR = os.path.join(ROOT_DIR, "models")


# Create necessary directories
for directory in [DATA_DIR, RAW_DATA_DIR, PREPARED_DATA_DIR, MODEL_DIR]:
    create_directory_if_not_exists(directory)

# Define paths to data files
DATA_PATH_RAW = os.path.join(RAW_DATA_DIR, "raw.csv")
DATA_PATH_PREPARED = os.path.join(PREPARED_DATA_DIR, "prepared.csv")


USECOLS = [
    # origin
    "AppId",
    "dtstart",
    "birthday",
    "sex",
    "citizenshipid",
    "martialid",
    "dependents",
    "sitename",
    "DOC",
    "averagemonthlyincome",
    "requested_amount",
    # nbki
    "Days_since_last_credit",
    "Max_overdue",
    # equifax
    "Nb_delays_90plus_ever_eq",
    "CH_length_eq",
    "S_hare_active_credit",
    # megafon/mail
    "Score",
    "MatchingLevel",
    "LIFETIMEBINValueId",
    "INTEGRALSCOREValueId",
    "60_6mob",
]
