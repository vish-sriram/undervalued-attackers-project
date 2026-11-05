"""Fetch match-level xG for the historical team fixtures, with resumable cache."""
import concurrent.futures
import json
import time
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent
BASE = ROOT/'data/player_history/fotmob'
OUT = BASE/'matches'
OUT.mkdir(exist_ok=True)
tasks = json.loads((BASE/'required_matches.json').read_text())

def fetch(game):
    gid = str(game['id'])
    path = OUT/f'{gid}.json'
    url = f'https://www.fotmob.com/api/data/matchDetails?matchId={gid}'
    for attempt in range(3):
        try:
            if path.exists():
                match = json.loads(path.read_text())
            else:
                response = requests.get(url,timeout=45)
                response.raise_for_status()
                match = response.json()
            if str(match['general']['matchId']) != gid:
                raise ValueError('Wrong match returned')
            expected = game['status']['utcTime'][:10]
            if match['general']['matchTimeUTCDate'][:10] != expected:
                raise ValueError('Match-date mismatch')
            if not path.exists():
                temp=path.with_suffix('.tmp')
                temp.write_text(json.dumps(match,ensure_ascii=False))
                temp.replace(path)
                time.sleep(.5)
            return {'game_id':gid,'state':'available','url':url}
        except Exception as exc:
            if attempt<2:
                time.sleep(3*(attempt+1))
            else:
                return {'game_id':gid,'state':'failed','error':str(exc),'url':url}

results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    for result in executor.map(fetch,tasks):
        results.append(result)
        if len(results)%50==0 or len(results)==len(tasks):
            status={'expected':len(tasks),'attempted':len(results),
                    'available':sum(x['state']=='available' for x in results),
                    'complete':len(results)==len(tasks),'results':results}
            path=BASE/'match_collection_status.json'
            temp=path.with_suffix('.tmp');temp.write_text(json.dumps(status));temp.replace(path)
            print(json.dumps({k:v for k,v in status.items() if k!='results'}),flush=True)
