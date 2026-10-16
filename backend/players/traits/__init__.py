"""Per-position player trait ratings.

``TRAITS`` is keyed by ``(name, position_group)`` so a trait that several
position groups share (a tackling rate, say) registers one spec per group
without colliding. The position modules fill it as they are imported.
"""

from dataclasses import dataclass

MODEL_VERSION = "cfb_position_traits_v1"


@dataclass(frozen=True)
class TraitSpec:
    name: str
    position_group: str
    kind: str  # "counted" or "inferred"
    aggregate: str = "rate"  # "rate" | "quantile20" | "share_of_games" | "latest"
    higher_is_better: bool | None = True


TRAITS: dict[tuple[str, str], TraitSpec] = {}


def register(spec: TraitSpec) -> TraitSpec:
    key = (spec.name, spec.position_group)
    if key in TRAITS:
        raise ValueError(
            f"trait {spec.name!r} is already registered for {spec.position_group}"
        )
    TRAITS[key] = spec
    return spec
