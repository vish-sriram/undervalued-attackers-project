"""Exercise the existing SoccerData/WhoScored path for one missing season."""
import json
from datetime import datetime, timezone
from pathlib import Path

import soccerdata as sd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data/player_history'
OUT.mkdir(parents=True, exist_ok=True)
status = {'league': 'ENG-Premier League', 'season': '2425', 'started_at': datetime.now(timezone.utc).isoformat()}
ws = None
try:
    ws = sd.WhoScored(leagues='ENG-Premier League', seasons=2024, headless=True)
    schedule = ws.read_schedule(force_cache=True).reset_index()
    schedule.to_csv(OUT / 'probe_schedule.csv', index=False)
    status.update(status='schedule_available', scheduled_matches=len(schedule))
    played = schedule.dropna(subset=['home_score', 'away_score'])
    if len(played):
        game_id = int(played.iloc[0]['game_id'])
        events = ws.read_events(match_id=game_id, output_fmt='spadl', force_cache=True, on_error='raise')
        status.update(status='events_available', probe_game_id=game_id, probe_actions=len(events))
        events.to_parquet(OUT / 'probe_actions.parquet', index=False)
except Exception as exc:
    status.update(status='failed', error_type=type(exc).__name__, error=str(exc))
finally:
    if ws is not None and hasattr(ws, '_driver'):
        ws._driver.quit()
    status['finished_at'] = datetime.now(timezone.utc).isoformat()
    (OUT / 'scraper_probe.json').write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2), flush=True)
