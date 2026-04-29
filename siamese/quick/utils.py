from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Define base data directory
DATA_DIR = PROJECT_ROOT / "data"

# Input directories
LOSS_VIDEOS = DATA_DIR / "videos_loss"
NORMAL_VIDEOS = DATA_DIR / "videos_normal"
GROUND_TRUTH_CSV = DATA_DIR / "ground_truth.csv"

# Output directories
OUT_DIR = DATA_DIR / "extracted_pairs"
INIT_DIR = OUT_DIR / "initial"
CURR_DIR = OUT_DIR / "current"
PAIRS_CSV_PATH = OUT_DIR / "pairs_labels.csv"

# Helper function to ensure output directories exist before extraction
def create_output_dirs():
    INIT_DIR.mkdir(parents=True, exist_ok=True)
    CURR_DIR.mkdir(parents=True, exist_ok=True)