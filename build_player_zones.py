"""Tag each action with a pitch zone and summarize thirds and channels.

Thirds run from the player's own goal toward the goal he is attacking.
Channels run from his right touchline to his left. Threat shares are the
percentage of that player's total xT from each third and each channel.
Pass shares are the percentage of his SPADL passes from each third and channel.
"""

import sys
from pathlib import Path

import duckdb

DB_PATH = Path(__file__).resolve().parent / "premier_league_2025_26.duckdb"
EVENT_TABLES = ("match_events", "team_events", "player_events")
PASS_TYPE_ID = 0

ZONE_SQL = """
    third = CASE
        WHEN start_x < 35 THEN 'defensive'
        WHEN start_x < 70 THEN 'middle'
        WHEN start_x IS NULL THEN NULL
        ELSE 'attacking'
    END,
    channel = CASE
        WHEN start_y < 13.6 THEN 'right'
        WHEN start_y < 27.2 THEN 'right_half'
        WHEN start_y < 40.8 THEN 'centre'
        WHEN start_y < 54.4 THEN 'left_half'
        WHEN start_y IS NULL THEN NULL
        ELSE 'left'
    END
"""

SHARE_COLUMNS = [
    "xt_pct_defensive_third",
    "xt_pct_middle_third",
    "xt_pct_attacking_third",
    "xt_pct_right",
    "xt_pct_right_half",
    "xt_pct_centre",
    "xt_pct_left_half",
    "xt_pct_left",
    "pass_pct_defensive_third",
    "pass_pct_middle_third",
    "pass_pct_attacking_third",
    "pass_pct_right",
    "pass_pct_right_half",
    "pass_pct_centre",
    "pass_pct_left_half",
    "pass_pct_left",
]


def tag_events(con):
    for table in EVENT_TABLES:
        con.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS third VARCHAR")
        con.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS channel VARCHAR")
        con.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS zone VARCHAR")
        con.execute(f"UPDATE {table} SET {ZONE_SQL}").fetchall()
        con.execute(
            f"""
            UPDATE {table}
            SET zone = third || '_' || channel
            WHERE third IS NOT NULL AND channel IS NOT NULL
            """
        ).fetchall()


def threat_shares(con):
    """Percentage of each player's total xT in each third and channel."""
    return con.execute(
        """
        WITH tagged AS (
            SELECT
                CAST(player_id AS BIGINT) AS player_id,
                third,
                channel,
                COALESCE(xT, 0) AS xt
            FROM player_events
            WHERE player_id IS NOT NULL AND third IS NOT NULL AND channel IS NOT NULL
        ),
        totals AS (
            SELECT player_id, SUM(xt) AS xt_total
            FROM tagged
            GROUP BY 1
        )
        SELECT
            t.player_id,
            100 * SUM(CASE WHEN t.third = 'defensive' THEN t.xt ELSE 0 END) / NULLIF(s.xt_total, 0) AS xt_pct_defensive_third,
            100 * SUM(CASE WHEN t.third = 'middle' THEN t.xt ELSE 0 END) / NULLIF(s.xt_total, 0) AS xt_pct_middle_third,
            100 * SUM(CASE WHEN t.third = 'attacking' THEN t.xt ELSE 0 END) / NULLIF(s.xt_total, 0) AS xt_pct_attacking_third,
            100 * SUM(CASE WHEN t.channel = 'right' THEN t.xt ELSE 0 END) / NULLIF(s.xt_total, 0) AS xt_pct_right,
            100 * SUM(CASE WHEN t.channel = 'right_half' THEN t.xt ELSE 0 END) / NULLIF(s.xt_total, 0) AS xt_pct_right_half,
            100 * SUM(CASE WHEN t.channel = 'centre' THEN t.xt ELSE 0 END) / NULLIF(s.xt_total, 0) AS xt_pct_centre,
            100 * SUM(CASE WHEN t.channel = 'left_half' THEN t.xt ELSE 0 END) / NULLIF(s.xt_total, 0) AS xt_pct_left_half,
            100 * SUM(CASE WHEN t.channel = 'left' THEN t.xt ELSE 0 END) / NULLIF(s.xt_total, 0) AS xt_pct_left
        FROM tagged AS t
        JOIN totals AS s ON t.player_id = s.player_id
        GROUP BY t.player_id, s.xt_total
        """
    ).fetchdf()


def pass_shares(con):
    """Percentage of each player's passes in each third and channel."""
    return con.execute(
        f"""
        WITH passes AS (
            SELECT
                CAST(player_id AS BIGINT) AS player_id,
                third,
                channel
            FROM player_events
            WHERE type_id = {PASS_TYPE_ID}
              AND player_id IS NOT NULL
              AND third IS NOT NULL
              AND channel IS NOT NULL
        ),
        totals AS (
            SELECT player_id, COUNT(*) AS pass_total
            FROM passes
            GROUP BY 1
        )
        SELECT
            t.player_id,
            100.0 * SUM(CASE WHEN t.third = 'defensive' THEN 1 ELSE 0 END) / s.pass_total AS pass_pct_defensive_third,
            100.0 * SUM(CASE WHEN t.third = 'middle' THEN 1 ELSE 0 END) / s.pass_total AS pass_pct_middle_third,
            100.0 * SUM(CASE WHEN t.third = 'attacking' THEN 1 ELSE 0 END) / s.pass_total AS pass_pct_attacking_third,
            100.0 * SUM(CASE WHEN t.channel = 'right' THEN 1 ELSE 0 END) / s.pass_total AS pass_pct_right,
            100.0 * SUM(CASE WHEN t.channel = 'right_half' THEN 1 ELSE 0 END) / s.pass_total AS pass_pct_right_half,
            100.0 * SUM(CASE WHEN t.channel = 'centre' THEN 1 ELSE 0 END) / s.pass_total AS pass_pct_centre,
            100.0 * SUM(CASE WHEN t.channel = 'left_half' THEN 1 ELSE 0 END) / s.pass_total AS pass_pct_left_half,
            100.0 * SUM(CASE WHEN t.channel = 'left' THEN 1 ELSE 0 END) / s.pass_total AS pass_pct_left
        FROM passes AS t
        JOIN totals AS s ON t.player_id = s.player_id
        GROUP BY t.player_id, s.pass_total
        """
    ).fetchdf()


def main():
    if not DB_PATH.exists():
        sys.exit(f"Error: database not found at {DB_PATH}")

    try:
        con = duckdb.connect(str(DB_PATH))
        tag_events(con)
        shares = threat_shares(con).merge(pass_shares(con), on="player_id", how="outer")
        players = con.execute("SELECT * FROM player_season_stats").fetchdf()
        players = players.drop(columns=[column for column in SHARE_COLUMNS if column in players.columns])
        table = players.merge(shares, on="player_id", how="left")
        con.register("player_zone_df", table)
        con.execute("CREATE OR REPLACE TABLE player_season_stats AS SELECT * FROM player_zone_df")
        tagged = con.execute(
            "SELECT COUNT(*) FILTER (WHERE zone IS NOT NULL), COUNT(*) FROM player_events"
        ).fetchall()[0]
        con.close()
    except Exception as exc:
        sys.exit(f"Error: {exc}")

    print(f"Database: {DB_PATH}")
    print(f"player_events tagged: {tagged[0]} of {tagged[1]}")
    print(
        f"player_season_stats: {len(table)} players, "
        f"{int(table['xt_pct_attacking_third'].notna().sum())} with threat shares, "
        f"{int(table['pass_pct_attacking_third'].notna().sum())} with pass shares"
    )


if __name__ == "__main__":
    main()
