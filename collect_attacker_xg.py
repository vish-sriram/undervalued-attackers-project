"""Fetch separately attributed Understat xG through SoccerData."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('SOCCERDATA_DIR', str(ROOT / 'data/soccerdata_runtime'))
import soccerdata as sd

OUT = ROOT / 'data/player_history/understat'
OUT.mkdir(parents=True, exist_ok=True)
LEAGUES = ['ENG-Premier League', 'ESP-La Liga', 'GER-Bundesliga', 'ITA-Serie A', 'FRA-Ligue 1']
status = []
for year in [2023, 2024, 2025]:
    for league in LEAGUES:
        record = {'league': league, 'season_start': year}
        try:
            reader = sd.Understat(leagues=league, seasons=year)
            players = reader.read_player_season_stats(force_cache=True).reset_index()
            players.to_csv(OUT / f'{league}_{year}_players.csv', index=False)
            schedule = reader.read_schedule(force_cache=True).reset_index()
            schedule.to_csv(OUT / f'{league}_{year}_schedule.csv', index=False)
            record.update(status='available', players=len(players), scheduled_matches=len(schedule))
        except Exception as exc:
            record.update(status='failed', error=str(exc), error_type=type(exc).__name__)
        status.append(record)
        (OUT / 'status.json').write_text(json.dumps(status, indent=2))
        print(json.dumps(record), flush=True)
