"""Identify actual historical teams from FotMob season minute records."""
import json
from pathlib import Path
from collections import defaultdict
from attacker_history_metrics import normalize

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data/player_history'
pool = json.loads((DATA / 'target_pool.json').read_text())
verified={(r['player_id'],r['fotmob_player_id']) for r in json.loads((DATA/'verified_fotmob_identities.json').read_text()) if r['verified']}
plan, crosswalk = defaultdict(set), []
for path in (DATA / 'fotmob').glob('*_mins_played.json'):
    year = int(path.name.split('_')[-3])
    league = path.name.rsplit(f'_{year}_', 1)[0]
    stats = json.loads(path.read_text())
    for block in stats['TopLists']:
        for row in block['StatList']:
            matches = [p for p in pool if normalize(p['player']) == normalize(row['ParticipantName'])
                       and year == int(p['window'][-4:])-1
                       and (p['player_id'],row['ParticiantId']) in verified]
            if len(matches) != 1:
                continue
            target = matches[0]
            crosswalk.append({'player_id': target['player_id'], 'name': target['player'],
                'season_start': year, 'league': league, 'fotmob_player_id': row['ParticiantId'],
                'fotmob_name': row['ParticipantName'], 'fotmob_team_id': row['TeamId'],
                'team': row['TeamName'], 'minutes': row['StatValue'],
                'identity_method': 'Name/spelling alias plus exact date-of-birth match across primary profiles'})
            plan[f'{league}_{year}'].add(row['TeamName'])
(DATA / 'historical_identity_crosswalk.json').write_text(json.dumps(crosswalk, indent=2, ensure_ascii=False))
(DATA / 'required_team_seasons.json').write_text(json.dumps({k: sorted(v) for k,v in plan.items()}, indent=2))
games={}
for row in crosswalk:
    league=json.loads((DATA/'fotmob'/f"{row['league']}_{row['season_start']}_league.json").read_text())
    for game in league['fixtures']['allMatches']:
        if game['status'].get('finished') and not game['status'].get('cancelled') and str(row['fotmob_team_id']) in [str(game['home']['id']),str(game['away']['id'])]:
            games[game['id']]={**game,'league':row['league'],'season_start':row['season_start']}
(DATA/'fotmob/required_matches.json').write_text(json.dumps(list(games.values())))
print(json.dumps({'player_league_seasons': len(crosswalk), 'datasets': len(plan),
                 'team_seasons': sum(map(len, plan.values())), 'required_matches':len(games)}, indent=2))
