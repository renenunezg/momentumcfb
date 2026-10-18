"""Compact, self-describing serving snapshot; full research detail stays offline."""

TEAM_COLUMNS = [
    "id",
    "team",
    "conference",
    "ap_rank",
    "wins",
    "losses",
    "conference_wins",
    "conference_losses",
    "playoff",
    "bye",
    "champion",
    "conference_champion",
    "conference_title_game",
    "expected_wins",
    "expected_losses",
    "selection_sensitivity",
    "dependence_tiebreak_sensitivity",
]
GAME_COLUMNS = [
    "id",
    "home_id",
    "away_id",
    "date",
    "home_win_probability",
    "neutral",
    "provisional",
]


def compact(payload):
    """Store repeated field names once and each remaining matchup once."""
    result = {k: v for k, v in payload.items() if k != "teams"}
    result["schema_version"] = 2
    result["team_columns"] = TEAM_COLUMNS
    result["teams"] = [[t[k] for k in TEAM_COLUMNS] for t in payload["teams"]]
    games = {}
    names = {str(t["id"]): t["team"] for t in payload["teams"]}
    for t in payload["teams"]:
        for g in t["remaining"]:
            opponent = g["opponent_id"]
            names.setdefault(str(opponent), g["opponent"])
            home, away = (t["id"], opponent) if g["home"] else (opponent, t["id"])
            p = g["win_probability"] if g["home"] else 1 - g["win_probability"]
            row = [
                g["game_id"],
                home,
                away,
                g["date"],
                p,
                g["neutral"],
                g["provisional"],
            ]
            if g["game_id"] in games:
                old = games[g["game_id"]]
                assert old[:4] == row[:4] and old[5:] == row[5:]
                assert abs(old[4] - p) < 1e-12
            else:
                games[g["game_id"]] = row
    result["game_columns"] = GAME_COLUMNS
    result["games"] = list(games.values())
    # Only names not already present in the team table need another entry.
    result["opponent_names"] = {
        k: v
        for k, v in names.items()
        if int(k) not in {t["id"] for t in payload["teams"]}
    }
    return result
