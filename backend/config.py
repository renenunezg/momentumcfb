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

# Program facts the team model and the player models both read.
POWER_CONFERENCES = frozenset({"SEC", "Big Ten", "Big 12", "ACC"})
# The Pac-12 that re-formed in 2026 is a Group of Five league.
PAC_12_POWER_THROUGH = 2025
# Service academies do not sign rated recruiting classes, so the talent
# composite scores them near zero (Army 23.66 against a median near 650 in
# 2025). Their talent is treated as missing: on the 22 academy games of the
# 2022 through 2025 Week 1 and 2 slates that cut the preseason prior's distance
# to the closing margin from 14.7 to 6.3 points and its margin MAE from 16.4 to
# 14.2.
UNRATED_TALENT_TEAMS = frozenset({"Air Force", "Army", "Navy"})
