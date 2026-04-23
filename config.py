# CNN
DATA_FOLDER = "testdata"
DATA_FILE = "vin17_processed.csv"
SEQ_LENGTH = 10
FEATURE_COLS = ["totalodometer","chargestatus","totalvoltage","totalcurrent","minvoltagebattery","maxvoltagebattery","mintemperaturevalue","maxtemperaturevalue"]
TARGET_COLS = ["soc"]
INPUT_CHANNELS = len(FEATURE_COLS)

from loguru import logger
import sys
logger.remove()

logger.add(
    sys.stdout,
    format="<level>{level}</level> | {name}:{function}:{line} | <level>{message}</level>",
    colorize=True
)