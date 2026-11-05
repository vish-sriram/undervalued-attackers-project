"""Collect public historical FotMob pages and their linked statistics JSON.

FotMob is the explicitly separate xG source; no xG is inferred from shots.
"""
import json
import time
from pathlib import Path
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data/player_history/fotmob'
OUT.mkdir(parents=True, exist_ok=True)
LEAGUES = {
    'ENG-Premier League': (47, 'premier-league'),
    'GER-Bundesliga': (54, 'bundesliga'),
    'FRA-Ligue 1': (53, 'ligue-1'),
    'ESP-La Liga': (87, 'laliga'),
    'ITA-Serie A': (55, 'serie-a'),
    'NED-Eredivisie': (57, 'eredivisie'),
    'ENG-Championship': (48, 'championship'),
    'ESP-La Liga 2': (140, 'laliga2'),
    'FRA-Ligue 2': (110, 'ligue-2'),
    'GER-2. Bundesliga': (146, '2-bundesliga'),
    'SUI-Super League': (69, 'super-league'),
}
STAT_NAMES = {'expected_goals', 'expected_goals_per_90', 'mins_played', 'total_scoring_att'}

def get(url, path):
    if path.exists():
        return json.loads(path.read_text())
    response = requests.get(url, timeout=45)
    response.raise_for_status()
    if 'text/html' in response.headers.get('Content-Type', ''):
        script = BeautifulSoup(response.text, 'html.parser').find('script', id='__NEXT_DATA__')
        if script is None:
            raise ValueError('No page data')
        value = json.loads(script.string)['props']['pageProps']
    else:
        value = response.json()
    path.write_text(json.dumps(value, ensure_ascii=False))
    time.sleep(0.5)
    return value

status = []
for year in [2023, 2024, 2025]:
    season = f'{year}/{year+1}'
    for league, (lid, slug) in LEAGUES.items():
        key = f'{league}_{year}'
        url = f'https://www.fotmob.com/leagues/{lid}/overview/{slug}?season={year}%2F{year+1}'
        record = {'league': league, 'season': season, 'source_url': url}
        try:
            data = get(url, OUT / f'{key}_league.json')
            if data['details']['selectedSeason'] != season:
                raise ValueError(f"Wrong season returned: {data['details']['selectedSeason']}")
            if data['details']['id'] != lid:
                raise ValueError('Wrong competition returned')
            record['competition_name'] = data['details']['name']
            record['stats'] = {}
            for stat in data.get('stats', {}).get('players', []):
                if stat['name'] not in STAT_NAMES:
                    continue
                stat_url = stat['fetchAllUrl']
                values = get(stat_url, OUT / f"{key}_{stat['name']}.json")
                record['stats'][stat['name']] = {'url': stat_url, 'available': True}
            record['state'] = 'available'
        except Exception as exc:
            record.update(state='failed', error=str(exc))
        status.append(record)
        (OUT / 'collection_status.json').write_text(json.dumps(status, indent=2))
        print(json.dumps(record), flush=True)
