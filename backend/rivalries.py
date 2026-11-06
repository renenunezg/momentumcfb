"""Named FBS rivalries for the rivalry road-favourite pick rule.

The list was written from general football knowledge before any results were
looked at, then frozen; it is not selected on outcomes. Names are CFBD school
names. Pairs between an FBS and an FCS program are left out because the rule
only covers FBS versus FBS games.
"""

_PAIRS = """
Alabama|Auburn; Ohio State|Michigan; Army|Navy; Oklahoma|Texas; UCLA|USC;
Notre Dame|USC; Florida|Georgia; Florida|Florida State; Georgia|Georgia Tech;
Clemson|South Carolina; Oregon|Oregon State; Washington|Washington State;
California|Stanford; Arizona|Arizona State; BYU|Utah; Colorado|Colorado State;
Michigan|Michigan State; Minnesota|Wisconsin; Iowa|Iowa State; Iowa|Minnesota;
Iowa|Nebraska; Indiana|Purdue; Illinois|Northwestern; Kansas|Kansas State;
Oklahoma|Oklahoma State; Texas|Texas A&M; Arkansas|LSU; Mississippi State|Ole Miss;
Kentucky|Louisville; Tennessee|Vanderbilt; Alabama|Tennessee; Auburn|Georgia;
Virginia|Virginia Tech; NC State|North Carolina; Duke|North Carolina;
Florida State|Miami; Pittsburgh|West Virginia; Air Force|Navy; Air Force|Army;
Navy|Notre Dame; Michigan State|Notre Dame; Houston|Rice; SMU|TCU; Baylor|TCU;
Nevada|UNLV; New Mexico|New Mexico State; New Mexico State|UTEP;
Bowling Green|Toledo; Miami (OH)|Ohio; Central Michigan|Western Michigan;
Colorado State|Wyoming; Southern Miss|Tulane; App State|Georgia Southern;
Arkansas|Missouri; LSU|Ole Miss; Alabama|LSU; Florida|Tennessee; Kansas|Missouri;
Boise State|Fresno State; Fresno State|San José State; Hawai'i|San José State;
Memphis|Tulane; East Carolina|Marshall; Cincinnati|Louisville;
Boston College|Syracuse; Maryland|Penn State; Penn State|Pittsburgh; TCU|Texas;
Oklahoma State|Texas Tech; Baylor|Texas; Kansas State|Missouri;
Louisiana|Louisiana Tech; Louisiana Tech|Southern Miss;
Middle Tennessee|Western Kentucky; Marshall|Ohio; Ball State|Northern Illinois;
Akron|Kent State; Buffalo|Kent State; Central Michigan|Eastern Michigan;
Utah|Utah State; BYU|Utah State; Florida Atlantic|Florida International;
South Alabama|Troy; Arkansas State|Louisiana; Georgia Southern|Georgia State;
App State|Coastal Carolina; Army|Notre Dame; Florida|Miami; Clemson|Georgia;
LSU|Texas A&M; Maryland|Rutgers
"""

RIVALRY_PAIRS = frozenset(
    frozenset(name.strip() for name in pair.split("|"))
    for pair in _PAIRS.replace("\n", " ").split(";")
)


def is_rivalry(home_team: str, away_team: str) -> bool:
    return frozenset((home_team, away_team)) in RIVALRY_PAIRS
