from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

DATA_DIR = REPO_ROOT / "backend" / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

CFBD_BASE_URL = "https://api.collegefootballdata.com"

SEASONS = list(range(2019, 2026))
EVAL_SEASONS = list(range(2021, 2026))
MAX_REGULAR_WEEK = 16

FCS_LABEL = "FCS"
