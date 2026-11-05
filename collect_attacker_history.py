"""Resume SoccerData/WhoScored collection for pre-transfer league seasons.

Uses the same reader and cache as scrape_premier_league_spadl.py. Raw match
JSON is retained so receptions, minutes, injuries and xT can be audited.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
STATE = ROOT / 'data/player_history/collection'
RUNTIME = ROOT / 'data/soccerdata_runtime'
os.environ.setdefault('SOCCERDATA_DIR', str(RUNTIME))
config = RUNTIME / 'config'
config.mkdir(parents=True, exist_ok=True)
custom = config / 'league_dict.json'
extra = json.loads(custom.read_text()) if custom.exists() else {}
for key, ws_name in {
    'NED-Eredivisie': 'Netherlands - Eredivisie',
    'ENG-Championship': 'England - Championship',
    'ESP-La Liga 2': 'Spain - Segunda División',
    'FRA-Ligue 2': 'France - Ligue 2',
    'GER-2. Bundesliga': 'Germany - 2. Bundesliga',
    'SUI-Super League': 'Switzerland - Super League',
}.items():
    extra[key] = {'WhoScored': ws_name, 'season_start': 'Aug', 'season_end': 'May'}
custom.write_text(json.dumps(extra))

LEAGUES = ['ENG-Premier League', 'GER-Bundesliga', 'FRA-Ligue 1',
           'ESP-La Liga', 'ITA-Serie A', 'NED-Eredivisie']

def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, default=str))
    temp.replace(path)

def collect(league, season):
    import soccerdata as sd
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    STATE.mkdir(parents=True, exist_ok=True)
    code = f'{season % 100:02d}{(season + 1) % 100:02d}'
    key = f'{league}_{code}'
    status_path = STATE / f'{key}.json'
    cache = Path.home() / 'soccerdata/data/WhoScored'
    status = {'league': league, 'season_start': season, 'season': code,
              'state': 'collecting_schedule', 'started_at': datetime.now(timezone.utc).isoformat(),
              'errors': [], 'available_game_ids': []}
    save(status_path, status)
    reader = None
    try:
        reader = sd.WhoScored(leagues=league, seasons=season, headless=True, data_dir=cache)
        schedule = reader.read_schedule(force_cache=True).reset_index()
        schedule.to_csv(STATE / f'{key}_schedule.csv', index=False)
        played = schedule.dropna(subset=['home_score', 'away_score']).drop_duplicates('game_id')
        from attacker_history_metrics import team_key
        plan = json.loads((ROOT / 'data/player_history/required_team_seasons.json').read_text())
        required_names = plan.get(f'{league}_{season}', [])
        required = {team_key(t) for t in required_names}
        status['league_scheduled_games'] = len(played)
        status['required_teams'] = required_names
        all_names = set(played['home_team']) | set(played['away_team'])
        status['unmatched_team_names'] = [t for t in required_names if team_key(t) not in {team_key(n) for n in all_names}]
        if (league, season) != ('ENG-Premier League', 2024):
            played = played[played['home_team'].map(team_key).isin(required) |
                            played['away_team'].map(team_key).isin(required)]
        status.update(state='collecting_events', expected_games=len(played))
        save(status_path, status)
        for n, match in enumerate(played.to_dict('records'), 1):
            gid = int(match['game_id'])
            path = cache / 'events' / key / f'{gid}.json'
            try:
                valid = False
                if path.exists():
                    data = json.loads(path.read_text())
                    valid = bool(data and data.get('events') and data.get('home') and data.get('away'))
                if not valid:
                    reader.read_events(match_id=gid, output_fmt=None, force_cache=True, on_error='raise')
                    data = json.loads(path.read_text()) if path.exists() else None
                    valid = bool(data and data.get('events') and data.get('home') and data.get('away'))
                if not valid:
                    raise ValueError('No usable raw match event data returned')
                status['available_game_ids'].append(gid)
            except Exception as exc:
                status['errors'].append({'game_id': gid, 'error': str(exc)})
            status.update(attempted_games=n, available_games=len(status['available_game_ids']))
            save(status_path, status)
            if n % 25 == 0:
                print(json.dumps({'dataset': key, 'available': status['available_games'], 'expected': len(played)}), flush=True)
        status['state'] = 'complete' if not status['errors'] and not status['unmatched_team_names'] else 'partial'
    except Exception as exc:
        status.update(state='failed', error=str(exc), error_type=type(exc).__name__)
    finally:
        if reader is not None and hasattr(reader, '_driver'):
            reader._driver.quit()
        status['finished_at'] = datetime.now(timezone.utc).isoformat()
        save(status_path, status)
    return status

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--leagues', nargs='+')
    parser.add_argument('--seasons', nargs='+', type=int)
    args = parser.parse_args()
    plan = json.loads((ROOT / 'data/player_history/required_team_seasons.json').read_text())
    tasks = [(key.rsplit('_', 1)[0], int(key.rsplit('_', 1)[1])) for key in plan]
    if args.leagues:
        tasks=[t for t in tasks if t[0] in args.leagues]
    if args.seasons:
        tasks=[t for t in tasks if t[1] in args.seasons]
    tasks.sort(key=lambda x: (x != ('ENG-Premier League', 2024), x[1], x[0]))
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(collect, *task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            s = future.result()
            print(json.dumps({k: s.get(k) for k in ['league', 'season', 'state', 'available_games', 'expected_games', 'error']}), flush=True)

if __name__ == '__main__':
    main()
