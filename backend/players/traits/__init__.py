"""Per-position player trait ratings.

``TRAITS`` is keyed by trait name. A trait that several position groups share
(a havoc rate across DL, LB, and DB, say) is one spec listing every group in
``position_groups``. The position modules fill it as they are imported.
"""

from dataclasses import dataclass

MODEL_VERSION = "cfb_position_traits_v1"
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


# Importing a position module registers its traits, so the registry is complete
# whenever the package is imported.
from backend.players.traits import defense, line, offense, punting  # noqa: E402, F401
