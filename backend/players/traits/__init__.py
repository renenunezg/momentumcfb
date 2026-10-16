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


def register(spec: TraitSpec) -> TraitSpec:
    if spec.name in TRAITS:
        raise ValueError(f"trait {spec.name!r} is already registered")
    TRAITS[spec.name] = spec
    return spec


# Importing a position module registers its traits, so the registry is complete
# whenever the package is imported.
from backend.players.traits import offense  # noqa: E402, F401
