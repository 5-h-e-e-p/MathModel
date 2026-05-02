PRED_HORIZON = 360
BEST_MODEL_PATH = f"horizon={PRED_HORIZON}/best_model.pth"
X_SCALER_PATH = f"horizon={PRED_HORIZON}/x_scaler.joblib"
Y_SCALER_PATH = f"horizon={PRED_HORIZON}/y_scaler.joblib"

DATA_FOLDER = "data"
DATA_FILE = r"data\vin17_processed.csv"
SEQ_LENGTH = 10
FEATURE_COLS = ["totalodometer","chargestatus","totalvoltage","totalcurrent","minvoltagebattery","maxvoltagebattery","mintemperaturevalue","maxtemperaturevalue"]
TARGET_COLS = ["soc"]
INPUT_CHANNELS = len(FEATURE_COLS)
TRAIN_RATE = 0.7
VAL_RATE = 0.15
TEST_RATE = 0.15

from loguru import logger
import sys
logger.remove()

logger.add(
    sys.stdout,
    format="<level>{level}</level> | {name}:{function}:{line} | <level>{message}</level>",
    colorize=True
)