"""The lifecycle of one row in ``cfb.recommendations``.

A row's state is not stored. It is read from three facts: the decision
(``status``), the settlement (``outcome``) and which side of kickoff the clock
is on. ``_FACTS`` is the one place that reading is written down; the Python
classifier and the publisher's SQL predicates are both derived from it.

    open no-play ──kickoff──> locked no-play ──final score──> passed
    open pick ────kickoff──> locked pick ────final score──> settled
                                           └─moneyline tie─> void
    a moved kickoff voids a pick at any time and passes a locked no-play

An open no-play row is re-decided in place by the publisher until kickoff. A
published pick is never re-decided here. The closing line is a field of a
settled row, recorded once, not a state. ``cfb.protect_recommendation``
enforces the same rules independently in the database.
"""

from enum import StrEnum

import pandas as pd

SETTLED_OUTCOMES = ("win", "loss", "push")


class PickState(StrEnum):
    OPEN_NO_PLAY = "open_no_play"
    OPEN_PICK = "open_pick"
    LOCKED_NO_PLAY = "locked_no_play"
    LOCKED_PICK = "locked_pick"
    PASSED = "passed"
    SETTLED = "settled"
    VOID = "void"


class PickEvent(StrEnum):
    KICKOFF_MOVED = "kickoff_moved"
    FINAL_SCORE = "final_score"
    MONEYLINE_TIE = "moneyline_tie"


# state: (status, outcomes, kickoff still ahead). None means the fact does not
# distinguish the state. The trigger already refuses a settled no-play and a
# passed pick, so the terminal states do not restate the status.
_FACTS: dict[PickState, tuple[str | None, tuple[str, ...], bool | None]] = {
    PickState.OPEN_NO_PLAY: ("no_play", ("pending",), True),
    PickState.OPEN_PICK: ("recommended", ("pending",), True),
    PickState.LOCKED_NO_PLAY: ("no_play", ("pending",), False),
    PickState.LOCKED_PICK: ("recommended", ("pending",), False),
    PickState.PASSED: (None, ("no_play",), None),
    PickState.SETTLED: (None, SETTLED_OUTCOMES, None),
    PickState.VOID: (None, ("void",), None),
}
PENDING_PICK = (PickState.OPEN_PICK, PickState.LOCKED_PICK)

# Every settlement the grader may record. A pair that is absent leaves the
# row as it is: an open no-play whose kickoff moved is simply re-decided.
SETTLEMENTS: dict[tuple[PickState, PickEvent], PickState] = {
    (PickState.OPEN_PICK, PickEvent.KICKOFF_MOVED): PickState.VOID,
    (PickState.LOCKED_PICK, PickEvent.KICKOFF_MOVED): PickState.VOID,
    (PickState.LOCKED_NO_PLAY, PickEvent.KICKOFF_MOVED): PickState.PASSED,
    (PickState.LOCKED_NO_PLAY, PickEvent.FINAL_SCORE): PickState.PASSED,
    (PickState.LOCKED_PICK, PickEvent.FINAL_SCORE): PickState.SETTLED,
    (PickState.LOCKED_PICK, PickEvent.MONEYLINE_TIE): PickState.VOID,
}


def pick_state(status: str, outcome: str, start_date, now) -> PickState:
    """The state of one stored row as of ``now``."""
    # A missing kickoff is never ahead, as the grader has always read it.
    ahead = bool(pd.Timestamp(now) < pd.to_datetime(start_date, utc=True))
    for state, (decision, outcomes, kickoff_ahead) in _FACTS.items():
        if (
            decision in (None, status)
            and outcome in outcomes
            and kickoff_ahead in (None, ahead)
        ):
            return state
    raise ValueError(f"no pick state for status {status!r}, outcome {outcome!r}")


def where_state(alias: str, *states: PickState) -> str:
    """SQL predicate for rows of ``alias`` in any of ``states``.

    A decision named on both sides of kickoff needs no clock comparison.
    """
    clauses = []
    for state in states:
        decision, outcomes, ahead = _FACTS[state]
        both_sides = ahead is not None and any(
            _FACTS[other][:2] == (decision, outcomes)
            and _FACTS[other][2] is (not ahead)
            for other in states
        )
        if both_sides and not ahead:
            continue
        parts = []
        if decision is not None:
            parts.append(f"{alias}.status = '{decision}'")
        parts.append(
            f"{alias}.outcome = '{outcomes[0]}'"
            if len(outcomes) == 1
            else f"{alias}.outcome IN ({', '.join(repr(o) for o in outcomes)})"
        )
        if ahead is not None and not both_sides:
            parts.append(
                f"{alias}.start_date {'>' if ahead else '<='} clock_timestamp()"
            )
        clauses.append(" AND ".join(parts))
    return clauses[0] if len(clauses) == 1 else f"({' OR '.join(clauses)})"
