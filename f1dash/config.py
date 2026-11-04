"""Colours, design tokens, documented modelling constants and the cache path."""

import os
import tempfile
from pathlib import Path

# ── OpenF1 ────────────────────────────────────────────────────────────────
OPENF1_BASE_URL = "https://api.openf1.org/v1"
FIRST_SEASON = 2023  # OpenF1 has no data before 2023.
MAX_REQUESTS_PER_SECOND = 3  # Free-tier limits published by OpenF1.
MAX_REQUESTS_PER_MINUTE = 30
REQUEST_TIMEOUT_S = 45
MAX_RETRIES = 3
DISK_CACHE_TTL_S = 6 * 3600  # A race can gain corrections for a few hours after the flag.

# ── Modelling constants ───────────────────────────────────────────────────
SEED = 42
FUEL_EFFECT = 0.03  # s/lap gained as fuel burns. Added back to isolate tyre wear.
MIN_STINT_LAPS = 6  # Fewest clean laps for a degradation fit.
BOOTSTRAP_RESAMPLES = 500
RECENT_LAPS = 10  # Laps used for current pace in the overtake simulation.
MIN_RECENT_LAPS = 5
N_SIMULATIONS = 2000
RHO_MAX = 0.8  # Upper clip on lap-to-lap correlation.
PREDICTION_HORIZON = 3  # Laps ahead for the position predictor.
TEST_FRACTION = 0.2  # Share of laps in the predictor's test period.
MIN_DATASET_ROWS = 50
SECTOR_SUM_TOLERANCE_S = 0.5  # Largest allowed gap between sector sum and lap time.
QUICK_LAP_RATIO = 1.07  # Laps slower than this multiple of the stint median are not clean.

# ── Design tokens (mirrored as CSS variables in assets/style.css) ─────────
BG = "#0A0A0A"
SURFACE = "#131313"
SURFACE_2 = "#1B1B1B"
BORDER = "#262626"
TEXT = "#F5F5F5"
TEXT_DIM = "#9CA3AF"
SOFT = "#E8002D"
MEDIUM = "#FFD600"
HARD = "#F2F2F2"
INTER = "#39B54A"
WET = "#0072CE"

COMPOUND_COLORS = {
    "SOFT": SOFT,
    "MEDIUM": MEDIUM,
    "HARD": HARD,
    "INTERMEDIATE": INTER,
    "WET": WET,
}
COMPOUND_LETTERS = {"SOFT": "S", "MEDIUM": "M", "HARD": "H", "INTERMEDIATE": "I", "WET": "W"}
COMPOUND_CODES = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4}
FALLBACK_TEAM_COLOR = "#9CA3AF"


def cache_dir() -> Path:
    """Return the OpenF1 response cache directory, creating it if needed."""
    path = Path(tempfile.gettempdir()) / "f1dash_openf1_cache"
    os.makedirs(path, exist_ok=True)
    return path
