"""Translate depth-chart roles to draft groups without treating packages as formations."""

POSITION_NAMES = {
    "QB": "Quarterback",
    "RB": "Running back",
    "FB": "Fullback",
    "WR": "Wide receiver",
    "TE": "Tight end",
    "LT": "Left tackle",
    "LG": "Left guard",
    "C": "Center",
    "RG": "Right guard",
    "RT": "Right tackle",
    "LDE": "Left defensive end",
    "RDE": "Right defensive end",
    "LDT": "Left defensive tackle",
    "RDT": "Right defensive tackle",
    "NT": "Nose tackle",
    "LILB": "Left inside linebacker",
    "RILB": "Right inside linebacker",
    "MLB": "Middle linebacker",
    "SLB": "Strongside linebacker",
    "WLB": "Weakside linebacker",
    "LCB": "Left cornerback",
    "RCB": "Right cornerback",
    "NB": "Nickelback / slot corner",
    "FS": "Free safety",
    "SS": "Strong safety",
    "H": "Holder",
    "KR": "Kick returner",
    "LS": "Long snapper",
    "P": "Punter",
    "PK": "Placekicker",
    "PR": "Punt returner",
}


def draft_role(code, package):
    if code not in POSITION_NAMES:
        raise ValueError(f"Unreviewed depth-chart position: {package}: {code}")
    if code in ("LDE", "RDE", "SLB", "WLB"):
        if package not in ("Base 3-4 D", "Base 4-3 D"):
            raise ValueError(f"Unreviewed defensive package: {package}")
        edge = (
            code in ("SLB", "WLB") if package == "Base 3-4 D" else code.endswith("DE")
        )
        return (
            ("EDGE", "Front seven")
            if edge
            else ("DT" if code.endswith("DE") else "LB", "Front seven")
        )
    for codes, role, group in [
        (("LT", "RT"), "OT", "Offensive line"),
        (("LG", "C", "RG"), "IOL", "Offensive line"),
        (("LDT", "RDT", "NT"), "DT", "Front seven"),
        (("LILB", "RILB", "MLB"), "LB", "Front seven"),
        (("LCB", "RCB"), "CB", "Secondary"),
        (("FS", "SS"), "S", "Secondary"),
        (("NB",), "CB", "Nickel option"),
        (("FB",), "RB", "Fullback option"),
        (("H", "KR", "LS", "P", "PK", "PR"), None, "Specialists"),
    ]:
        if code in codes:
            return role, group
    return code, "Backfield" if code in ("QB", "RB") else "Receivers"


def normalize_depth(rows):
    for row in rows:
        role, group = draft_role(row["pos_abb"], row["pos_grp"])
        row.update(
            draft_position=role,
            display_group=group,
            position_name=POSITION_NAMES[row["pos_abb"]],
        )
    return rows
