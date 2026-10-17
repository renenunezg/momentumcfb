"""Intended receivers for incomplete passes.

The CFBD play-stats feed carries a ``Target`` row for most incompletions, but
not all of them. Where it is missing, the receiver is usually named in the play
text ("pass incomplete to Jon Smith"), so it is recovered by matching that name
to the offense's roster.
"""

import re
import unicodedata

import pandas as pd

INCOMPLETION = "Pass Incompletion"
SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}
TARGET_COLUMNS = ["play_id", "athlete_id", "source"]
_PENALTY = re.compile(r"\bpenalty\b.*", re.IGNORECASE | re.DOTALL)
_RECEIVER = re.compile(
    r"(?i:pass incomplete to )(?P<name>[A-Z][^,;(]+?)(?: for\b| broken up|"
    r" intercepted|,|\(|$)"
)


def normalize_name(value: str) -> str:
    """Casefold, strip accents and punctuation, drop generational suffixes."""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-z0-9\s]", "", text.casefold())
    return " ".join(word for word in text.split() if word not in SUFFIXES)


def athlete_ids(values: pd.Series) -> pd.Series:
    """String ids; an integer-valued float (4.0) becomes "4" and nulls stay null."""
    return values.astype("string").str.replace(r"\.0$", "", regex=True)


_ids = athlete_ids


def _receiver_name(play_text: object) -> str | None:
    text = _PENALTY.sub("", play_text) if isinstance(play_text, str) else ""
    match = _RECEIVER.search(text)
    return match.group("name").strip() if match else None


def recover_targets(
    plays: pd.DataFrame, play_stats: pd.DataFrame, roster: pd.DataFrame
) -> pd.DataFrame:
    """One intended receiver per play: the feed's Target row, else the play text.

    Only incompletions without a feed target are parsed, and a name shared by two
    players on the offense's roster is skipped rather than guessed.
    """
    incomplete = plays[plays["play_type"].eq(INCOMPLETION)]

    feed = play_stats[play_stats["stat_type"].eq("Target")]
    feed = feed.assign(
        play_id=feed["play_id"].astype(str), athlete_id=_ids(feed["athlete_id"])
    )
    feed = feed.dropna(subset=["athlete_id"]).astype({"athlete_id": object})
    feed = feed[feed["play_id"].isin(set(plays["id"].astype(str)))]
    feed = feed.drop_duplicates("play_id")[["play_id", "athlete_id"]].assign(
        source="feed"
    )

    names = (
        roster.assign(
            key=(
                roster["first_name"].fillna("") + " " + roster["last_name"].fillna("")
            ).map(normalize_name),
            athlete_id=_ids(roster["id"]),
        )
        .drop_duplicates(["team", "key", "athlete_id"])
        .groupby(["team", "key"])["athlete_id"]
        .agg(list)
    )
    unique = {key: ids[0] for key, ids in names.items() if len(ids) == 1}

    rows = []
    fed = set(feed["play_id"])
    for play in incomplete.itertuples(index=False):
        play_id = str(play.id)
        if play_id in fed:
            continue
        name = _receiver_name(play.play_text)
        athlete_id = unique.get((play.offense, normalize_name(name))) if name else None
        if athlete_id:
            rows.append(
                {"play_id": play_id, "athlete_id": athlete_id, "source": "text"}
            )
    text = pd.DataFrame(rows, columns=TARGET_COLUMNS)
    return pd.concat([feed[TARGET_COLUMNS], text], ignore_index=True)


def target_coverage(plays: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Per season: incompletions, and how many have a feed, text, or no target."""
    incomplete = plays[plays["play_type"].eq(INCOMPLETION)]
    source = targets.assign(play_id=targets["play_id"].astype(str)).set_index(
        "play_id"
    )["source"]
    covered = incomplete["id"].astype(str).map(source)
    frame = pd.DataFrame(
        {"season": incomplete["season"].to_numpy(), "source": covered.to_numpy()}
    )
    grouped = frame.groupby("season")["source"]
    out = pd.DataFrame(
        {
            "incompletions": grouped.size(),
            "feed": grouped.apply(lambda s: int(s.eq("feed").sum())),
            "text": grouped.apply(lambda s: int(s.eq("text").sum())),
        }
    )
    out["uncovered"] = out["incompletions"] - out["feed"] - out["text"]
    return out.reset_index()
