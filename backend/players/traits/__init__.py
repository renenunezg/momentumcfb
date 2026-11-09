"""Per-position player trait ratings.

``TRAITS`` is keyed by trait name. A trait that several position groups share
(a havoc rate across DL, LB, and DB, say) is one spec listing every group in
``position_groups``. The position modules fill it as they are imported.
"""

from dataclasses import dataclass

# v1 was the 2026-10-16 fit on CFBD's pre-revision EPA; v1.1 is the 2026-10-17 refit
# of the same code on the revised EPA, the model that has been served since. The
# string tells served rows apart; it is checked, never derived, on load. Its
# weights were fit on credit before untagged interceptions were charged to the
# passer (2026-10-28); the ``epa.credit`` trait is z-scored within each week's
# pool, so the level shift cancels and only the rank moves reach the grade.
# v2 (2026-11-07) is the same recipe refit on player value v6 credit with the
# receiver room trait (``rec.room``) available to WR and TE.
# v3 (2026-11-09) adds the program talent trait (``team.talent``) at QB, RB and DB,
# leaves DL trait z-scores uncapped, and pulls the early-season composite toward
# last season's final grade at the groups the model carries prior terms for.
MODEL_VERSION = "cfb_position_traits_v3"
AGGREGATES = frozenset({"rate", "quantile20", "share_of_games", "latest"})


@dataclass(frozen=True)
class TraitSpec:
    name: str
    position_groups: tuple[str, ...]
    kind: str  # "counted" or "inferred"
    aggregate: str = "rate"  # "rate" | "quantile20" | "share_of_games" | "latest"
    higher_is_better: bool | None = True

    def __post_init__(self) -> None:
        if self.aggregate not in AGGREGATES:
            raise ValueError(
                f"trait {self.name!r} has unknown aggregate {self.aggregate!r}"
            )


TRAITS: dict[str, TraitSpec] = {}
# Traits the grade reads that no builder emits: they are added to the snapshots
# from other models (``epa.credit`` from player value v4). Kept apart from
# ``TRAITS`` so the builders' registry stays exactly what they produce.
DERIVED_TRAITS: dict[str, TraitSpec] = {}
EPA_TRAIT = "epa.credit"
TALENT_TRAIT = "team.talent"


def register(spec: TraitSpec) -> TraitSpec:
    if spec.name in TRAITS:
        raise ValueError(f"trait {spec.name!r} is already registered")
    TRAITS[spec.name] = spec
    return spec


def all_traits() -> dict[str, TraitSpec]:
    """``TRAITS`` and ``DERIVED_TRAITS`` together."""
    return {**TRAITS, **DERIVED_TRAITS}


# v4 ``shrunk_per_game`` through the week, the player's EPA credit, as a trait. It has
# no OL value. Per game it is the game's adjusted EPA, so a ``rate`` over games with
# denominator 1 is its per-game mean for split-half reliability.
DERIVED_TRAITS[EPA_TRAIT] = TraitSpec(
    EPA_TRAIT, ("QB", "RB", "WR", "TE", "DL", "LB", "DB"), "counted"
)

# The talent composite of the player's program that season, a level prior: at equal
# production a player on a more talented roster did better the next season at these
# groups on the 2020-2024 development pairs (QB, RB, DB), and no better elsewhere.
# It is known before the season, has no per-game value and so no reliability; the
# grade weights it only through the fitted step.
DERIVED_TRAITS[TALENT_TRAIT] = TraitSpec(
    TALENT_TRAIT, ("QB", "RB", "DB"), "inferred", aggregate="latest"
)


# Importing a position module registers its traits, so the registry is complete
# whenever the package is imported.
from backend.players.traits import defense, line, offense, punting  # noqa: E402, F401
