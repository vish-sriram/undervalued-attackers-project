"""Build a season table of WhoScored per-player stats, as per-90 rates.

The numbers come from the player stats object already stored on each
downloaded 2025/26 match file. Counting stats are summed, then divided by
minutes per 90. Rating and the success percentages are season rates.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import duckdb
import pandas as pd

DB_PATH = Path(__file__).resolve().parent / "premier_league_2025_26.duckdb"
EVENTS_DIR = (
    Path.home() / "soccerdata/data/WhoScored/events/ENG-Premier League_2526"
)

# These series are percentages or a running rating, not event counts.
RATE_STATS = {
    "aerialSuccess",
    "dribbleSuccess",
    "passSuccess",
    "ratings",
    "tackleSuccess",
    "throwInAccuracy",
}

RATE_FROM_COUNTS = {
    "aerialSuccess": ("aerialsWon", "aerialsTotal"),
    "dribbleSuccess": ("dribblesWon", "dribblesAttempted"),
    "passSuccess": ("passesAccurate", "passesTotal"),
    "tackleSuccess": ("tackleSuccessful", "tacklesTotal"),
    "throwInAccuracy": ("throwInsAccurate", "throwInsTotal"),
}


def sending_off(event):
    labels = {
        (qualifier.get("type") or {}).get("displayName")
        for qualifier in event.get("qualifiers") or []
    }
    return "Red" in labels or "SecondYellow" in labels


def match_length(match):
    if match.get("maxMinute"):
        return int(match["maxMinute"])
    ends = match.get("periodEndMinutes") or {}
    if "2" in ends:
        return int(ends["2"])
    return 90


def minutes_by_player(match):
    """Minutes from the lineup plus substitutions and sending-offs."""
    end = match_length(match)
    state = {}
    meta = {}
    for side in ("home", "away"):
        team = match[side]["name"]
        for player in match[side]["players"]:
            player_id = int(player["playerId"])
            state[player_id] = {
                "on": 0 if player.get("isFirstEleven") else None,
                "minutes": 0,
            }
            meta[player_id] = {
                "name": player.get("name"),
                "team": team,
                "position": player.get("position"),
                "stats": player.get("stats") or {},
            }

    for event in match["events"]:
        kind = (event.get("type") or {}).get("displayName")
        player_id = event.get("playerId")
        minute = event.get("minute")
        if player_id is None or minute is None or int(player_id) not in state:
            continue
        player_id = int(player_id)
        minute = int(minute)
        if kind == "SubstitutionOff" or (kind == "Card" and sending_off(event)):
            if state[player_id]["on"] is not None:
                state[player_id]["minutes"] += max(minute - state[player_id]["on"], 0)
                state[player_id]["on"] = None
        elif kind == "SubstitutionOn" and state[player_id]["on"] is None:
            state[player_id]["on"] = minute

    for player_id, status in state.items():
        if status["on"] is not None:
            status["minutes"] += max(end - status["on"], 0)
    return state, meta


def final_rating(stats):
    ratings = stats.get("ratings")
    if not ratings:
        return None
    last_minute = max(ratings, key=lambda minute: int(minute))
    return float(ratings[last_minute])


def main():
    if not EVENTS_DIR.exists():
        sys.exit(f"Error: match files not found at {EVENTS_DIR}")

    totals = defaultdict(float)
    rating_weight = defaultdict(float)
    minutes = defaultdict(float)
    appearances = defaultdict(int)
    names = {}
    team_minutes = defaultdict(lambda: defaultdict(float))
    position_minutes = defaultdict(lambda: defaultdict(float))

    try:
        paths = sorted(EVENTS_DIR.glob("*.json"))
        if not paths:
            raise LookupError("no match files found")

        for path in paths:
            match = json.loads(path.read_text())
            played, meta = minutes_by_player(match)
            for player_id, status in played.items():
                played_minutes = status["minutes"]
                if played_minutes <= 0:
                    continue
                info = meta[player_id]
                minutes[player_id] += played_minutes
                appearances[player_id] += 1
                names[player_id] = info["name"]
                team_minutes[player_id][info["team"]] += played_minutes
                if info["position"]:
                    position_minutes[player_id][info["position"]] += played_minutes
                for stat, series in info["stats"].items():
                    if stat in RATE_STATS or not isinstance(series, dict):
                        continue
                    totals[(player_id, stat)] += sum(float(value) for value in series.values())
                rating = final_rating(info["stats"])
                if rating is not None:
                    rating_weight[player_id] += rating * played_minutes
    except Exception as exc:
        sys.exit(f"Error: {exc}")

    count_stats = sorted({stat for (_, stat) in totals})
    rows = []
    for player_id, played_minutes in minutes.items():
        per90 = played_minutes / 90
        row = {
            "player_id": player_id,
            "player": names[player_id],
            "team": max(team_minutes[player_id], key=team_minutes[player_id].get),
            "position": (
                max(position_minutes[player_id], key=position_minutes[player_id].get)
                if position_minutes[player_id]
                else None
            ),
            "appearances": appearances[player_id],
            "minutes": played_minutes,
            "rating": (
                rating_weight[player_id] / played_minutes
                if rating_weight[player_id]
                else None
            ),
        }
        for stat in count_stats:
            row[f"{stat}_per90"] = totals[(player_id, stat)] / per90
        for stat, (made, attempted) in RATE_FROM_COUNTS.items():
            attempts = totals[(player_id, attempted)]
            row[stat] = 100 * totals[(player_id, made)] / attempts if attempts else None
        rows.append(row)

    table = pd.DataFrame(rows).sort_values(["team", "player"]).reset_index(drop=True)
    try:
        con = duckdb.connect(str(DB_PATH))
        con.register("player_season_df", table)
        con.execute(
            "CREATE OR REPLACE TABLE player_season_stats AS SELECT * FROM player_season_df"
        )
        con.execute("ALTER TABLE player_season_stats ADD COLUMN xT_per90 DOUBLE")
        con.execute(
            """
            UPDATE player_season_stats
            SET xT_per90 = COALESCE(e.xt_sum, 0) * 90.0 / player_season_stats.minutes
            FROM (
                SELECT CAST(player_id AS BIGINT) AS player_id, SUM(xT) AS xt_sum
                FROM player_events
                GROUP BY 1
            ) AS e
            WHERE player_season_stats.player_id = e.player_id
            """
        )
        con.execute(
            "UPDATE player_season_stats SET xT_per90 = 0 WHERE xT_per90 IS NULL"
        )
        con.close()
    except Exception as exc:
        sys.exit(f"Error: {exc}")

    print(f"Database: {DB_PATH}")
    print(f"player_season_stats: {len(table)} players, {table.shape[1]} columns")
    print(f"matches read: {len(paths)}")


if __name__ == "__main__":
    main()
