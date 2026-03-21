from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Config:
    MAIN_DATA_FILE: Path = Path("motor_data14-2018.csv")

    RAW_DIR: Path = Path("data/raw")
    PROCESSED_DIR: Path = Path("data/processed")
    MODELS_DIR: Path = Path("models")
    OUTPUTS_DIR: Path = Path("outputs")
    REPORTS_DIR: Path = Path("reports")

    TARGET_COL: str = "CLAIM_PAID"

    FEATURE_COLS: list = field(default_factory=lambda: [
        "SEX", "INSR_BEGIN", "INSR_END", "EFFECTIVE_YR", "INSR_TYPE",
        "INSURED_VALUE", "PREMIUM", "OBJECT_ID", "PROD_YEAR", "SEATS_NUM",
        "CARRYING_CAPACITY", "TYPE_VEHICLE", "CCM_TON", "MAKE", "USAGE"
    ])

    BATCH_SIZE: int = 10000

    MAX_MISSING_RATE: float = 0.6
    MAX_DUPLICATE_RATE: float = 0.2

    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    CV_FOLDS: int = 5

    LR_FIT_INTERCEPT: bool = True

    DT_MAX_DEPTH: int = 10
    DT_MIN_SAMPLES_SPLIT: int = 2
    DT_MIN_SAMPLES_LEAF: int = 1

    MLP_HIDDEN_LAYER_SIZES: tuple = (100, 50)
    MLP_MAX_ITER: int = 500
    MLP_ALPHA: float = 0.0001
    MLP_SOLVER: str = "adam"

CFG = Config()
