"""Resolve namesakes using dates of birth from both primary player profiles."""
import concurrent.futures
import json
from datetime import datetime
from pathlib import Path
import requests
from bs4 import BeautifulSoup

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data/player_history'
OUT=DATA/'fotmob/identity_profiles'
OUT.mkdir(exist_ok=True)
pool=json.loads((DATA/'target_pool.json').read_text())
births={p['player_id']:datetime.strptime(p['birth_text'][:10],'%d/%m/%Y').date().isoformat()
        for p in json.loads((DATA/'injury_histories.json').read_text()) if p.get('birth_text')}
rows=json.loads((DATA/'historical_identity_crosswalk.json').read_text())
tasks={(r['player_id'],r['fotmob_player_id']) for r in rows}

def check(task):
    tid,fid=task
    path=OUT/f'{fid}.json'
    try:
        if path.exists():
            data=json.loads(path.read_text())
        else:
            url=f'https://www.fotmob.com/players/{fid}'
            response=requests.get(url,timeout=40);response.raise_for_status()
            script=BeautifulSoup(response.text,'html.parser').find('script',id='__NEXT_DATA__')
            data=json.loads(script.string)['props']['pageProps']
            path.write_text(json.dumps(data,ensure_ascii=False))
        player=data['data'];dob=player['birthDate']['utcTime'][:10]
        return {'player_id':tid,'fotmob_player_id':fid,'fotmob_name':player['name'],
            'fotmob_birth_date':dob,'transfermarkt_birth_date':births.get(tid),
            'verified':dob==births.get(tid),'source_url':f'https://www.fotmob.com/players/{fid}'}
    except Exception as exc:
        return {'player_id':tid,'fotmob_player_id':fid,'verified':False,'error':str(exc)}

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    results=list(executor.map(check,sorted(tasks)))
(DATA/'verified_fotmob_identities.json').write_text(json.dumps(results,indent=2,ensure_ascii=False))
print(json.dumps({'checked':len(results),'verified':sum(r['verified'] for r in results),
                  'rejected':[r for r in results if not r['verified']]},indent=2))
