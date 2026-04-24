"""
team_name_mapping.py
Normalize team names across ESPN, American Soccer Analysis, and football-data.org.
MLS team names vary significantly between data providers.
"""

# ── Canonical names ────────────────────────────────────────────────────────────
# These are the names used everywhere in this codebase.
CANONICAL_TEAMS = {
    "Atlanta United",
    "Austin FC",
    "CF Montréal",
    "Charlotte FC",
    "Chicago Fire",
    "Colorado Rapids",
    "Columbus Crew",
    "D.C. United",
    "FC Cincinnati",
    "FC Dallas",
    "Houston Dynamo",
    "Inter Miami CF",
    "LA Galaxy",
    "LAFC",
    "Minnesota United",
    "Nashville SC",
    "New England Revolution",
    "New York City FC",
    "New York Red Bulls",
    "Orlando City",
    "Philadelphia Union",
    "Portland Timbers",
    "Real Salt Lake",
    "San Jose Earthquakes",
    "Seattle Sounders",
    "Sporting Kansas City",
    "St. Louis City SC",
    "Toronto FC",
    "Vancouver Whitecaps",
}

# ── Raw-name → canonical-name mapping ─────────────────────────────────────────
# Covers ESPN display names, ASA team names, football-data.org names, and common
# abbreviations / partial matches.
_NAME_MAP: dict[str, str] = {
    # Atlanta United
    "atlanta united fc": "Atlanta United",
    "atlanta united": "Atlanta United",
    "atl utd": "Atlanta United",
    "atl": "Atlanta United",
    # Austin FC
    "austin fc": "Austin FC",
    "aus": "Austin FC",
    # CF Montréal
    "cf montréal": "CF Montréal",
    "cf montreal": "CF Montréal",
    "montreal impact": "CF Montréal",
    "montréal impact": "CF Montréal",
    "mtl": "CF Montréal",
    # Charlotte FC
    "charlotte fc": "Charlotte FC",
    "clt": "Charlotte FC",
    # Chicago Fire
    "chicago fire fc": "Chicago Fire",
    "chicago fire": "Chicago Fire",
    "chi": "Chicago Fire",
    # Colorado Rapids
    "colorado rapids": "Colorado Rapids",
    "col": "Colorado Rapids",
    # Columbus Crew
    "columbus crew sc": "Columbus Crew",
    "columbus crew": "Columbus Crew",
    "clb": "Columbus Crew",
    "crew sc": "Columbus Crew",
    # D.C. United
    "d.c. united": "D.C. United",
    "dc united": "D.C. United",
    "d.c united": "D.C. United",
    "dcu": "D.C. United",
    # FC Cincinnati
    "fc cincinnati": "FC Cincinnati",
    "cin": "FC Cincinnati",
    # FC Dallas
    "fc dallas": "FC Dallas",
    "dal": "FC Dallas",
    # Houston Dynamo
    "houston dynamo fc": "Houston Dynamo",
    "houston dynamo": "Houston Dynamo",
    "hou": "Houston Dynamo",
    # Inter Miami CF
    "inter miami cf": "Inter Miami CF",
    "inter miami": "Inter Miami CF",
    "miami": "Inter Miami CF",
    "mia": "Inter Miami CF",
    # LA Galaxy
    "la galaxy": "LA Galaxy",
    "los angeles galaxy": "LA Galaxy",
    "lag": "LA Galaxy",
    "galaxy": "LA Galaxy",
    # LAFC
    "lafc": "LAFC",
    "los angeles fc": "LAFC",
    "la fc": "LAFC",
    # Minnesota United
    "minnesota united fc": "Minnesota United",
    "minnesota united": "Minnesota United",
    "min": "Minnesota United",
    # Nashville SC
    "nashville sc": "Nashville SC",
    "nsh": "Nashville SC",
    "nashville": "Nashville SC",
    # New England Revolution
    "new england revolution": "New England Revolution",
    "ne revs": "New England Revolution",
    "new england": "New England Revolution",
    "ne": "New England Revolution",
    "nre": "New England Revolution",
    # New York City FC
    "new york city fc": "New York City FC",
    "nyc fc": "New York City FC",
    "nycfc": "New York City FC",
    "nyc": "New York City FC",
    # New York Red Bulls
    "new york red bulls": "New York Red Bulls",
    "ny red bulls": "New York Red Bulls",
    "ny/nj metrostars": "New York Red Bulls",
    "red bulls": "New York Red Bulls",
    "rbny": "New York Red Bulls",
    "nyrb": "New York Red Bulls",
    "ny": "New York Red Bulls",
    # Orlando City
    "orlando city sc": "Orlando City",
    "orlando city": "Orlando City",
    "orl": "Orlando City",
    # Philadelphia Union
    "philadelphia union": "Philadelphia Union",
    "phi": "Philadelphia Union",
    # Portland Timbers
    "portland timbers": "Portland Timbers",
    "por": "Portland Timbers",
    "timbers": "Portland Timbers",
    # Real Salt Lake
    "real salt lake": "Real Salt Lake",
    "rsl": "Real Salt Lake",
    # San Jose Earthquakes
    "san jose earthquakes": "San Jose Earthquakes",
    "sj earthquakes": "San Jose Earthquakes",
    "sje": "San Jose Earthquakes",
    "san jose": "San Jose Earthquakes",
    # Seattle Sounders
    "seattle sounders fc": "Seattle Sounders",
    "seattle sounders": "Seattle Sounders",
    "sea": "Seattle Sounders",
    "sounders": "Seattle Sounders",
    # Sporting Kansas City
    "sporting kansas city": "Sporting Kansas City",
    "skc": "Sporting Kansas City",
    "sporting kc": "Sporting Kansas City",
    "kansas city": "Sporting Kansas City",
    # St. Louis City SC
    "st. louis city sc": "St. Louis City SC",
    "st louis city sc": "St. Louis City SC",
    "st. louis city": "St. Louis City SC",
    "stl": "St. Louis City SC",
    # Toronto FC
    "toronto fc": "Toronto FC",
    "tor": "Toronto FC",
    "toronto": "Toronto FC",
    # Vancouver Whitecaps
    "vancouver whitecaps fc": "Vancouver Whitecaps",
    "vancouver whitecaps": "Vancouver Whitecaps",
    "van": "Vancouver Whitecaps",
    "whitecaps": "Vancouver Whitecaps",
}


def normalize_team_name(raw_name: str) -> str:
    """Return the canonical team name for a raw string from any data source.

    Falls back to the original (stripped) string if no match is found so the
    caller always gets *something* back rather than None.
    """
    if not raw_name:
        return raw_name
    key = raw_name.strip().lower()
    return _NAME_MAP.get(key, raw_name.strip())


def is_known_team(name: str) -> bool:
    """Return True if *name* normalizes to a known canonical MLS team."""
    return normalize_team_name(name) in CANONICAL_TEAMS
