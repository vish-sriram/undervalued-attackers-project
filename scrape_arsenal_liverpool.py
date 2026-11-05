"""Scrape WhoScored events for the latest Arsenal vs Liverpool match."""

import sys

import soccerdata as sd

# 2025 -> 2025/26, 2026 -> 2026/27. Cover both so the latest fixture is found
# whether or not they have already met in the current season.
SEASONS = [2025, 2026]
TEAMS = ["Arsenal", "Liverpool"]

try:
    ws = sd.WhoScored(leagues="ENG-Premier League", seasons=SEASONS)
    schedule = ws.read_schedule()

    mask = schedule["home_team"].isin(TEAMS) & schedule["away_team"].isin(TEAMS)
    played = schedule.loc[mask].dropna(subset=["home_score", "away_score"])
    if played.empty:
        raise LookupError("No completed Arsenal vs Liverpool match found.")

    match = played.sort_values("date").iloc[-1]
    events = ws.read_events(match_id=int(match["game_id"]))
    if events is None or events.empty:
        raise LookupError(f"No events returned for game {int(match['game_id'])}.")
except Exception as exc:
    sys.exit(f"Error: {exc}")

print(
    f"{match['home_team']} {int(match['home_score'])}-{int(match['away_score'])} "
    f"{match['away_team']} ({match['date']})"
)
print(f"{len(events)} events")
print(events.head())
