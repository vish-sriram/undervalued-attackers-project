"""Dump 2025/26 Premier League SPADL events into a local DuckDB database.

SoccerData returns the SPADL actions. This script stores that frame as
match_events and does not build the team or player tables.
"""

import sys
from pathlib import Path

import duckdb
import soccerdata as sd

DB_PATH = Path(__file__).resolve().parent / "premier_league_2025_26.duckdb"
LEAGUE = "ENG-Premier League"
SEASON = 2025  # soccerdata reads this as 2025/26


def query_all(con, sql):
    """Run a query and read every row before the connection is used again."""
    return con.execute(sql).fetchall()


def saved_game_ids(con):
    exists = query_all(
        con,
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = 'match_events'
        """,
    )[0][0]
    if not exists:
        return set()
    rows = query_all(con, "SELECT DISTINCT game_id FROM match_events")
    return {int(game_id) for (game_id,) in rows}


def insert_events(con, events):
    con.register("events_df", events)
    exists = query_all(
        con,
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = 'match_events'
        """,
    )[0][0]
    if exists:
        query_all(con, "INSERT INTO match_events BY NAME SELECT * FROM events_df")
    else:
        query_all(con, "CREATE TABLE match_events AS SELECT * FROM events_df")
    con.unregister("events_df")


def main():
    try:
        ws = sd.WhoScored(leagues=LEAGUE, seasons=SEASON, headless=True)
        schedule = ws.read_schedule(force_cache=True).reset_index()
    except Exception as exc:
        sys.exit(f"Error: {exc}")

    for column in ("game_id", "home_team", "away_team", "home_score", "away_score"):
        if column not in schedule.columns:
            sys.exit(f"Error: schedule is missing column {column}")

    played = schedule.dropna(subset=["home_score", "away_score"]).copy()
    if played.empty:
        sys.exit("Error: no completed Premier League matches found for 2025/26.")

    con = duckdb.connect(str(DB_PATH))
    try:
        done = saved_game_ids(con)
        pending = played[~played["game_id"].astype(int).isin(done)]
        print(f"{len(played)} matches, {len(done)} already saved, {len(pending)} to scrape")

        for number, (_, match) in enumerate(pending.iterrows(), start=1):
            game_id = int(match["game_id"])
            label = (
                f"{match['home_team']} {int(match['home_score'])}-"
                f"{int(match['away_score'])} {match['away_team']}"
            )
            try:
                events = ws.read_events(
                    match_id=game_id,
                    output_fmt="spadl",
                    force_cache=True,
                    on_error="skip",
                )
                if events is None or "game_id" not in events.columns or events.empty:
                    raise LookupError("no SPADL events returned")
                insert_events(con, events)
            except Exception as exc:
                print(f"Error: {label} ({game_id}): {exc}")
                continue
            print(f"[{number}/{len(pending)}] {label}: {len(events)} events", flush=True)

        saved = saved_game_ids(con)
        if not saved:
            sys.exit("Error: no match events were saved.")
        row_count = query_all(con, "SELECT COUNT(*) FROM match_events")[0][0]
    except Exception as exc:
        sys.exit(f"Error: {exc}")
    finally:
        con.close()

    print(f"Database: {DB_PATH}")
    print(f"match_events: {row_count} rows from {len(saved)} games")


if __name__ == "__main__":
    main()
