"""Fit an expected threat model on the downloaded 2025/26 SPADL events.

WhoScored already records coordinates in the attacking direction of the
team on the ball. SoccerAction then mirrors the away team, which puts
their attacks in their own box. This script undoes that mirror before
fitting, and writes the corrected coordinates and xT values back.
"""

import json
import sys
from pathlib import Path

import duckdb
import pandas as pd
from socceraction.spadl import config as spadl
from socceraction.xthreat import ExpectedThreat

DB_PATH = Path(__file__).resolve().parent / "premier_league_2025_26.duckdb"
MODEL_PATH = Path(__file__).resolve().parent / "xt_model.json"
EVENTS_DIR = Path.home() / "soccerdata/data/WhoScored/events/ENG-Premier League_2526"
FIELD_LENGTH = spadl.field_length
FIELD_WIDTH = spadl.field_width
SHOT = spadl.actiontypes.index("shot")

COLUMNS = [
    "game_id",
    "original_event_id",
    "period_id",
    "time_seconds",
    "team_id",
    "player_id",
    "start_x",
    "end_x",
    "start_y",
    "end_y",
    "type_id",
    "result_id",
    "bodypart_id",
    "action_id",
]


def home_team_ids():
    homes = {}
    for path in EVENTS_DIR.glob("*.json"):
        match = json.loads(path.read_text())
        homes[int(path.stem)] = int(match["home"]["teamId"])
    return homes


def games_with_mirrored_away_attacks(events, homes):
    """Away shots sitting in the own half while home shots do not."""
    shots = events.loc[events["type_id"] == SHOT, ["game_id", "team_id", "start_x"]]
    mirrored = []
    for game_id, home_id in homes.items():
        game_shots = shots[shots["game_id"] == game_id]
        if game_shots.empty:
            continue
        home_x = game_shots.loc[game_shots["team_id"] == home_id, "start_x"].median()
        away_x = game_shots.loc[game_shots["team_id"] != home_id, "start_x"].median()
        if pd.notna(home_x) and pd.notna(away_x) and home_x > 60 and away_x < 45:
            mirrored.append(game_id)
    return mirrored


def unmirror_away_attacks(con, game_ids, homes):
    homes_df = pd.DataFrame(
        {"game_id": game_ids, "home_team_id": [homes[game_id] for game_id in game_ids]}
    )
    con.register("homes_df", homes_df)
    for table in ("match_events", "team_events", "player_events"):
        con.execute(
            f"""
            UPDATE {table}
            SET start_x = {FIELD_LENGTH} - start_x,
                end_x = {FIELD_LENGTH} - end_x,
                start_y = {FIELD_WIDTH} - start_y,
                end_y = {FIELD_WIDTH} - end_y
            FROM homes_df
            WHERE {table}.game_id = homes_df.game_id
              AND CAST({table}.team_id AS BIGINT) != homes_df.home_team_id
            """
        ).fetchall()
    con.unregister("homes_df")


def write_xt(con, model):
    events = con.execute(
        f"""
        SELECT game_id, action_id, type_id, result_id,
               start_x, start_y, end_x, end_y
        FROM player_events
        """
    ).fetchdf()
    scored = events.dropna(subset=["start_x", "start_y", "end_x", "end_y"]).copy()
    scored["xT"] = model.rate(scored)
    con.register("xt_df", scored[["game_id", "action_id", "xT"]])
    con.execute("UPDATE player_events SET xT = NULL").fetchall()
    con.execute(
        """
        UPDATE player_events
        SET xT = xt_df.xT
        FROM xt_df
        WHERE player_events.game_id = xt_df.game_id
          AND player_events.action_id = xt_df.action_id
        """
    ).fetchall()
    con.unregister("xt_df")
    has_column = con.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'player_season_stats' AND column_name = 'xT_per90'
        """
    ).fetchall()[0][0]
    if not has_column:
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
    ).fetchall()
    con.execute(
        "UPDATE player_season_stats SET xT_per90 = 0 WHERE xT_per90 IS NULL"
    ).fetchall()


def main():
    if not DB_PATH.exists():
        sys.exit(f"Error: database not found at {DB_PATH}")
    if not EVENTS_DIR.exists():
        sys.exit(f"Error: match files not found at {EVENTS_DIR}")

    try:
        homes = home_team_ids()
        con = duckdb.connect(str(DB_PATH))
        events = con.execute(f"SELECT {', '.join(COLUMNS)} FROM match_events").fetchdf()
        if events.empty:
            raise LookupError("match_events is empty")
        events["team_id"] = events["team_id"].astype("int64")
        events["game_id"] = events["game_id"].astype("int64")

        mirrored = games_with_mirrored_away_attacks(events, homes)
        if mirrored:
            unmirror_away_attacks(con, mirrored, homes)
            events = con.execute(f"SELECT {', '.join(COLUMNS)} FROM match_events").fetchdf()

        usable = events.dropna(subset=["start_x", "start_y", "end_x", "end_y"])
        if usable.empty:
            raise LookupError("no actions with complete coordinates")
        model = ExpectedThreat().fit(usable)
        model.save_model(str(MODEL_PATH))
        write_xt(con, model)
        con.close()
    except Exception as exc:
        sys.exit(f"Error: {exc}")

    print(f"Away-team coordinates corrected in {len(mirrored)} matches")
    print(f"Model: {MODEL_PATH}")
    print(f"Grid: {model.l} x {model.w}")
    print(f"Actions: {len(usable)}")
    print(f"xT range: {model.xT.min():.5f} to {model.xT.max():.5f}")


if __name__ == "__main__":
    main()
