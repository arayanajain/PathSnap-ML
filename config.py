"""Central configuration for PathSnap (from-scratch high-accuracy pipeline)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Dataset root containing `train/` (80 trips) and `student_test/` (10 trips)
# Prefers repo-local `data/Datasets` if populated, otherwise falls back to Downloads
_REPO_DATA_ROOT = PROJECT_ROOT / "data" / "Datasets"
_DOWNLOADS_DATA_ROOT = Path(r"C:\Users\araya\Downloads\ML_Hackathon\Datasets")
DATA_ROOT = _REPO_DATA_ROOT if (_REPO_DATA_ROOT / "train").exists() else _DOWNLOADS_DATA_ROOT
TRAIN_DIR = DATA_ROOT / "train"
STUDENT_TEST_DIR = DATA_ROOT / "student_test"
# Evaluator-testing data is strictly excluded from preprocessing, feature selection, and training.

# Output & cache directories
CACHE_DIR = PROJECT_ROOT / "cache"
RESULTS_DIR = PROJECT_ROOT / "results"
PREP_RESULTS_DIR = RESULTS_DIR / "preprocessing"
FS_RESULTS_DIR = RESULTS_DIR / "feature_selection"
MODEL_RESULTS_DIR = RESULTS_DIR / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Reproducibility & Cross-Validation
RANDOM_SEED = 42
N_CV_FOLDS = 5

# Candidate generation settings (justified on 80 training trips)
BBOX_BUFFER_M = 350.0        # Local road-network bounding box margin around GPS track
CANDIDATE_RADIUS_M = 120.0   # Search radius in meters around each GPS observation
MAX_CANDIDATES_K = 8         # Top-K closest candidate road segments per observation
ARC_SAMPLE_STEP_M = 40.0     # Densification step for indexing long segments in cKDTree
LARGE_GAP_SEC = 5.0          # GPS time gap threshold (from EDA: median dt=1s, extreme gaps up to 153k s)
STATIONARY_SPEED_MPS = 0.8   # Below this speed, instantaneous GPS bearing is noisy

# Feature selection settings (fit strictly on training partition)
CORR_THRESHOLD = 0.97        # Prune near-duplicate collinear features above |r| > 0.97
TOP_K_FEATURES = 22          # Target number of selected features after consensus ranking
