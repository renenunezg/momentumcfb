# Player value v4: opponent effects

## What changed

v3 fitted each week's per-play unit effects with a two-sided ridge anchored on
the previous season's rating, then blended that fit a second time toward the
same anchor with weight `games / (games + 3)`. Because the ridge already
weighs the season's games against the anchor, the second blend left midseason
effects about 90 percent previous-season rating: on the 2025 artifact the
week 8 effects correlated 0.86 with the preseason prior (slope 0.93) and even
the week 17 effects correlated 0.76. In 2026 that credited a Wake Forest
quarterback for facing Louisville's 2025 pass defense while the 2026 team
model already rated it well below average.

v4 drops the second blend and instead regresses the previous-season anchor
toward its subdivision mean before the ridge sees it. Two numbers define it:

| Setting | Value | Meaning |
| --- | --- | --- |
| `OPPONENT_PRIOR_REGRESSION` | 0.7 | Anchor is 0.7 of last season's per-play rating |
| `OPPONENT_PRIOR_SHARE` | 0.3 | Each unit's prior sd is 0.3 of the channel's game sd |

The opponent prior artifact is built exactly as under v3, so a frozen v3 prior
stays valid; `PRIOR_COMPATIBLE_VERSIONS` accepts both labels.

## How the numbers were chosen

Walk-forward on each season independently: the effects for week w are fitted
on completed games before w, and each week-w game's per-play rush and pass
value is predicted as league mean plus offense effect minus opposing defense
effect. Skill is one minus the play-weighted squared error over the
league-mean baseline, averaged over weeks 2 through 13. Development seasons
are 2020 through 2024; 2025 is the independent check. The 2026 season was
never scored.

Coarse grid, mean of the rush and pass channels:

| Variant | Dev 2020-2024 | Val 2025 |
| --- | --- | --- |
| v3 (share 0.3 with the second blend) | 0.0808 | 0.0796 |
| Prior only | 0.0326 | 0.0193 |
| In-season only, share 0.3 | 0.0673 | 0.0704 |
| Regress 1.0, share 0.3 | 0.0872 | 0.0880 |
| Regress 0.85, share 0.3 | 0.0900 | 0.0913 |
| **Regress 0.7, share 0.3** | **0.0908** | **0.0925** |
| Regress 0.5, share 0.3 | 0.0886 | 0.0908 |
| Regress 0.7, share 0.25 | 0.0898 | 0.0905 |
| Regress 0.7, share 0.4 | 0.0766 | 0.0796 |

The chosen setting beats v3 in every development season (2020: 0.0912 vs
0.0734; 2021: 0.0976 vs 0.0862; 2022: 0.0734 vs 0.0629; 2023: 0.0939 vs
0.0904; 2024: 0.0978 vs 0.0910) and in every week bucket of both channels.

A finer grid (regression 0.6 to 0.8 in steps of 0.05, share 0.27, 0.30,
0.33) is flat: the development optimum, regression 0.7 with share 0.27,
scores 0.0909 against 0.0908 for share 0.30, and 2025 prefers share 0.30.
Neither number carries more precision than its first decimal.

## What it does not fix

Per-play unit efficiency over a handful of games is noisy, so a strong prior
still predicts best; shares at or above 0.4 lose skill in every bucket. The
adjustment therefore still moves a quarterback by single-digit EPA over six
games. A player who leads the raw EPA table will usually still lead after
adjustment.
