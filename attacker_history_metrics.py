from collections import Counter
import math
import unicodedata
import numpy as np

NAME_ALIASES = {
    "bengannondoak": "bendoak",
    "jamiebynoegittens": "jamiegittens",
    "savinho": "savio",
    "strandlarsen": "jorgenstrandlarsen",
    "jeanmatteobahoya": "jeanmatteobahoya",
    "tommywatson": "tomwatson",
    "arnaudkalimuendomuinga": "arnaudkalimuendo",
    "yeremipino": "yeremypino",
    "emanuelemegha": "emmanuelemegha",
}


def normalize(name):
    value = "".join(
        c for c in unicodedata.normalize("NFKD", name.lower()) if c.isalnum()
    )
    return NAME_ALIASES.get(value, value)


TEAM_ALIASES = {
    "manchestercity": "mancity",
    "manchesterunited": "manutd",
    "tottenhamhotspur": "tottenham",
    "brightonhovealbion": "brighton",
    "wolverhamptonwanderers": "wolves",
    "nottinghamforest": "forest",
    "parissaintgermain": "psg",
    "bayer04leverkusen": "leverkusen",
    "bayerleverkusen": "leverkusen",
    "eintrachtfrankfurt": "frankfurt",
    "bayernmunchen": "bayern",
    "fcbayernmunich": "bayern",
    "ajaxamsterdam": "ajax",
    "ogcnice": "nice",
    "rcstrasbourgalsace": "strasbourg",
    "losclille": "lille",
    "asroma": "roma",
    "acmilan": "milan",
    "staderennaisfc": "rennes",
    "fctoulouse": "toulouse",
    "vflwolfsburg": "wolfsburg",
    "tsg1899hoffenheim": "hoffenheim",
    "1fcunionberlin": "unionberlin",
    "estactroyes": "troyes",
    "celta": "celtavigo",
    "celtadevigo": "celtavigo",
    "realbetisbalompie": "realbetis",
    "caosasuna": "osasuna",
    "athleticclub": "athleticbilbao",
    "rb": "rbleipzig",
    "brightonandhovealbion": "brighton",
    "ipswichtown": "ipswich",
    "newcastleunited": "newcastle",
    "westhamunited": "westham",
    "staderennais": "rennes",
    "borussiadortmund": "dortmund",
    "stuttgart": "vfbstuttgart",
}


def team_key(name):
    name = normalize(name)
    if name.endswith("fc"):
        name = name[:-2]
    if name.startswith("afc"):
        name = name[3:]
    return TEAM_ALIASES.get(name, name)


def possession_entries(events):
    counts = Counter()
    holder = None
    period = None
    for event in events:
        event_period = (event.get("period") or {}).get("value")
        if event_period not in (1, 2):
            continue
        if event_period != period:
            holder, period = None, event_period
        kind = (event.get("type") or {}).get("displayName")
        if kind in {"End", "OffsideGiven", "CornerAwarded"}:
            holder = None
            continue
        pid = event.get("playerId")
        if pid is None or not event.get("isTouch", False):
            continue
        actor = (int(event.get("teamId", -1)), int(pid))
        if actor != holder:
            counts[int(pid)] += 1
        holder = None if kind == "Goal" else actor
    return counts


def regulation_minutes(match):
    on_pitch = {}
    minutes = {}
    for side in ("home", "away"):
        for player in match[side]["players"]:
            player_id = int(player["playerId"])
            on_pitch[player_id] = 0.0 if player.get("isFirstEleven") else None
            minutes[player_id] = 0.0

    for event in match["events"]:
        player_id = event.get("playerId")
        period = (event.get("period") or {}).get("value")
        if player_id is None or int(player_id) not in minutes or period not in (1, 2):
            continue
        player_id = int(player_id)
        minute = float(event.get("minute", 0)) + float(event.get("second", 0)) / 60
        minute = min(45 if period == 1 else 90, minute)
        kind = (event.get("type") or {}).get("displayName")
        qualifiers = {
            (qualifier.get("type") or {}).get("displayName")
            for qualifier in event.get("qualifiers", [])
        }
        sent_off = kind == "Card" and bool(qualifiers & {"Red", "SecondYellow"})
        started = on_pitch[player_id]
        if (kind == "SubstitutionOff" or sent_off) and started is not None:
            minutes[player_id] += max(0.0, minute - started)
            on_pitch[player_id] = None
        elif kind == "SubstitutionOn" and started is None:
            on_pitch[player_id] = minute

    for player_id, started in on_pitch.items():
        if started is not None:
            minutes[player_id] += max(0.0, 90 - started)
    return minutes


def uncertainty(totals, minutes):
    totals = np.asarray(totals, dtype=float)
    minutes = np.asarray(minutes, dtype=float)
    valid = np.isfinite(totals) & np.isfinite(minutes) & (minutes > 0)
    totals, minutes = totals[valid], minutes[valid]
    if not len(totals):
        return {
            "rate_per90": None,
            "weighted_sd_per90": None,
            "se_per90": None,
            "effective_matches": 0.0,
        }

    weights = minutes / 90
    exposure = weights.sum()
    rates = totals / weights
    mean = float(totals.sum() / exposure)
    squared_weights = (weights * weights).sum()
    effective_matches = float(exposure**2 / squared_weights)
    variance_denominator = exposure - squared_weights / exposure
    sd = None
    if variance_denominator > 0:
        variance = (weights * (rates - mean) ** 2).sum() / variance_denominator
        sd = math.sqrt(float(variance))

    se = None
    matches = len(totals)
    if matches >= 5 and minutes.sum() >= 180:
        residuals = totals - mean * weights
        variance = matches / (matches - 1) * (residuals * residuals).sum()
        se = math.sqrt(float(variance)) / exposure
    return {
        "rate_per90": mean,
        "weighted_sd_per90": sd,
        "se_per90": se,
        "effective_matches": effective_matches,
    }


def risk_adjusted_rate(
    rate,
    minutes,
    prior_rate,
    se,
    match_sd,
    *,
    prior_minutes=900,
    uncertainty_weight=1.0,
    volatility_weight=0.0,
):
    reliability = minutes / (minutes + prior_minutes)
    center = reliability * rate + (1 - reliability) * prior_rate
    adjusted = None
    if se is not None and (not volatility_weight or match_sd is not None):
        penalty = uncertainty_weight * se + volatility_weight * (match_sd or 0)
        adjusted = center - penalty
    return {
        "reliability": reliability,
        "shrunken_rate": center,
        "risk_adjusted_rate": adjusted,
    }
