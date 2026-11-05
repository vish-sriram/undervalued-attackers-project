"""Build pre-transfer attacker features, retaining coverage and uncertainty.

Run again after collection advances. Source match files are never changed.
"""
import csv
import hashlib
import json
import re
import warnings
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from socceraction.data.opta.loader import _eventtypesdf
from socceraction.data.opta.parsers import WhoScoredParser
from socceraction.spadl.opta import convert_to_actions
from socceraction.xthreat import ExpectedThreat, load_model, get_successful_move_actions
from attacker_history_metrics import normalize, team_key, possession_entries, regulation_minutes, uncertainty

warnings.filterwarnings('ignore', category=FutureWarning)
ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data/player_history'
OUT = ROOT / 'outputs/attacker_history'
OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path.home() / 'soccerdata/data/WhoScored/events'
POOL = json.loads((DATA / 'target_pool.json').read_text())
VERIFIED={(r['player_id'],r['fotmob_player_id']) for r in json.loads((DATA/'verified_fotmob_identities.json').read_text()) if r['verified']}
BY_NAME = {normalize(p['player']): p for p in POOL}
assert len(BY_NAME) == len(POOL) == 71

def spadl(path, league, year, match):
    parser = WhoScoredParser(str(path), competition_id=league, season_id=str(year), game_id=int(path.stem))
    events = pd.DataFrame.from_dict(parser.extract_events(), orient='index').merge(_eventtypesdf, on='type_id', how='left').reset_index(drop=True)
    actions = convert_to_actions(events, home_team_id=int(match['home']['teamId']))
    away = actions['team_id'] != int(match['home']['teamId'])
    for col in ['start_x', 'end_x']:
        actions.loc[away, col] = 105 - actions.loc[away, col]
    for col in ['start_y', 'end_y']:
        actions.loc[away, col] = 68 - actions.loc[away, col]
    return actions

def reference_model():
    """Prefer a full pre-summer-2025 surface; never fit to future seasons."""
    path = DATA / 'xt_reference_2024_25.json'
    if path.exists():
        return load_model(str(path)), '2025-05-25', path
    files = list((CACHE / 'ENG-Premier League_2425').glob('*.json'))
    status_path=DATA/'collection/ENG-Premier League_2425.json'
    training_status=json.loads(status_path.read_text()) if status_path.exists() else {}
    if len(files) == 380 and training_status.get('state')=='complete' and training_status.get('available_games')==380:
        frames = []
        for f in files:
            m = json.loads(f.read_text())
            frames.append(spadl(f, 'ENG-Premier League', 2024, m))
        actions = pd.concat(frames, ignore_index=True).dropna(subset=['start_x','start_y','end_x','end_y'])
        model = ExpectedThreat().fit(actions)
        model.save_model(str(path))
        (DATA / 'xt_reference_metadata.json').write_text(json.dumps({
            'league': 'ENG-Premier League', 'season': '2024/25', 'games': len(files),
            'actions': len(actions), 'trained_through': '2025-05-25',
            'direction': 'Undo away-team mirror introduced by Opta converter',
        }, indent=2))
        return model, '2025-05-25', path
    # Existing model is valid as-of summer 2026 only. Earlier transfer rows stay null.
    return load_model(str(ROOT / 'xt_model.json')), '2026-05-24', ROOT / 'xt_model.json'

def write_csv(path, rows, fieldnames=None):
    keys = list(fieldnames) if fieldnames is not None else list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

def fotmob_stats():
    records = []
    for f in (DATA / 'fotmob').glob('*_mins_played.json'):
        year = int(f.name.split('_')[-3])
        league = f.name.rsplit(f'_{year}_', 1)[0]
        other = {}
        for stat in ['expected_goals', 'expected_goals_per_90']:
            p = f.parent / f'{league}_{year}_{stat}.json'
            if p.exists():
                for block in json.loads(p.read_text())['TopLists']:
                    for row in block['StatList']:
                        other.setdefault((row['ParticiantId'],row['TeamId']), {})[stat] = row['StatValue']
        page = json.loads((f.parent / f'{league}_{year}_league.json').read_text())
        fixtures = page['fixtures']['allMatches']
        for block in json.loads(f.read_text())['TopLists']:
            for r in block['StatList']:
                target = BY_NAME.get(normalize(r['ParticipantName']))
                if not target or year != int(target['window'][-4:])-1 or (target['player_id'],r['ParticiantId']) not in VERIFIED:
                    continue
                values = other.get((r['ParticiantId'],r['TeamId']), {})
                xg = values.get('expected_goals')
                minutes = r['StatValue']
                games = [g for g in fixtures if str(r['TeamId']) in [str(g['home']['id']),str(g['away']['id'])]
                         and g['status'].get('finished') and not g['status'].get('cancelled')]
                records.append({'player_id': target['player_id'], 'name': target['player'],
                    'league': league, 'season_start': year, 'team': r['TeamName'],
                    'fotmob_player_id': r['ParticiantId'], 'fotmob_team_id': r['TeamId'],
                    'xg_total': xg, 'xg_source_minutes': minutes,
                    'xg_per90': xg*90/minutes if xg is not None and minutes else None,
                    'source_published_xg_per90': values.get('expected_goals_per_90'),
                    'team_games': games, 'team_matches': len(games),
                    'xg_source': 'FotMob',
                    'xg_source_url': f"https://data.fotmob.com/stats/{page['details']['id']}/season/" +
                        next((s['fetchAllUrl'].split('/season/')[1] for s in page['stats']['players'] if s['name']=='expected_goals'), ''),
                })
    return records

def injury_episodes():
    result = defaultdict(list)
    sources = {}
    for name in ['injury_histories.json', 'injury_history_extra_pages.json']:
        for player in json.loads((DATA/name).read_text()):
            sources[player['player_id']] = bool(player['tables'])
            for table in player['tables']:
                if len(table['headers']) != 6:
                    continue
                for r in table['rows']:
                    cells = r['cells']
                    if len(cells) != 6:
                        continue
                    try:
                        start = datetime.strptime(cells[2], '%d/%m/%Y').date()
                        end = datetime.strptime(cells[3], '%d/%m/%Y').date()
                    except ValueError:
                        continue
                    clubs = [a.get('title') or a['text'] for a in r['links'] if '/verein/' in a['href']]
                    result[player['player_id']].append({'season': cells[0], 'injury': cells[1],
                        'start': start, 'end': end, 'clubs': clubs, 'reported_games_all_competitions': cells[5]})
    return result, sources

def fotmob_match_stats(source_seasons):
    """Retain zero-shot appearances only when match xG coverage is explicit."""
    identities = defaultdict(dict)
    for s in source_seasons:
        identities[(s['league'],s['season_start'])][int(s['fotmob_player_id'])] = s
    plan = {str(g['id']):g for g in json.loads((DATA/'fotmob/required_matches.json').read_text())}
    output=[]
    for path in (DATA/'fotmob/matches').glob('*.json'):
        game=plan.get(path.stem)
        if not game:
            continue
        try:
            match=json.loads(path.read_text())
            meta=match['general']
            day=meta['matchTimeUTCDate'][:10]
            coverage=meta.get('coverageLevel')=='xG'
            shots=(match['content'].get('shotmap') or {}).get('shots',[])
            for pid,p in (match['content'].get('playerStats') or {}).items():
                identity=identities[(game['league'],game['season_start'])].get(int(pid))
                if not identity:
                    continue
                target=next(t for t in POOL if t['player_id']==identity['player_id'])
                if day>=target['transfer_date']:
                    continue
                stats={v.get('key'):v.get('stat',{}).get('value')
                       for section in p.get('stats',[]) for v in section.get('stats',{}).values()}
                minutes=stats.get('minutes_played')
                if not isinstance(minutes,(int,float)) or minutes<=0:
                    continue
                own=[s for s in shots if s.get('playerId')==int(pid) and not s.get('isOwnGoal')]
                valid=coverage and all(s.get('expectedGoals') is not None for s in own)
                xg=sum(s['expectedGoals'] for s in own) if valid else None
                npxg=sum(s['expectedGoals'] for s in own if s.get('situation')!='Penalty') if valid else None
                output.append({'player_id':identity['player_id'],'name':target['player'],
                    'fotmob_player_id':int(pid),'game_id':path.stem,'date':day,
                    'league':game['league'],'season_start':game['season_start'],
                    'team':p['teamName'],'minutes':minutes,'xg':xg,'npxg':npxg,
                    'source_url':f'https://www.fotmob.com/api/data/matchDetails?matchId={path.stem}'})
        except (ValueError,KeyError,TypeError):
            continue
    assert len({(r['player_id'],r['game_id']) for r in output})==len(output)
    return output

def injury_percentage(pid, seasons, episodes, sources, match_rows):
    if not sources.get(pid):
        return None, None, None, 'No injury entries available; unknown rather than zero'
    total = missed = 0
    appeared = {(r['date'], team_key(r['team'])) for r in match_rows}
    for year in sorted({s['season_start'] for s in seasons}):
        stints = [s for s in seasons if s['season_start']==year]
        observed_clubs={team_key(g['team']) for g in match_rows if g['season_start']==year}
        if len({(s['league'],s['fotmob_team_id']) for s in stints}) != 1 or len(observed_clubs)>1:
            return None, None, None, 'Multiple teams within a season; registration dates needed for denominator'
        s = stints[0]
        games = s['team_games']
        if not games:
            return None, None, None, 'Team fixture denominator unavailable'
        total += len(games)
        for g in games:
            day = date.fromisoformat(g['status']['utcTime'][:10])
            if (day.isoformat(),team_key(s['team'])) in appeared:
                continue
            intervals = [e for e in episodes.get(pid, []) if e['start'] <= day <= e['end']
                and (not e['clubs'] or team_key(s['team']) in {team_key(c) for c in e['clubs']})
                and not re.search(r'^ill$|illness|corona|quarantine|suspension|personal reasons|not match fit|^fitness$', e['injury'], re.I)]
            if intervals:
                missed += 1
    return (100*missed/total if total else None), missed, total, 'Reported injury-date overlap with league fixtures; full team-season denominator; appearances excluded'

def main():
    model, trained_through, model_path = reference_model()
    model_hash = hashlib.sha256(model_path.read_bytes() + Path(__file__).read_bytes() +
        (ROOT/'attacker_history_metrics.py').read_bytes()).hexdigest()
    cache_path = DATA / 'processed_target_matches.json'
    stored = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    saved, errors = {}, []
    births = {}
    for p in json.loads((DATA / 'injury_histories.json').read_text()):
        if p.get('birth_text'):
            try:
                births[p['player_id']] = datetime.strptime(p['birth_text'][:10],'%d/%m/%Y').date()
            except ValueError:
                pass
    for directory in CACHE.iterdir():
        if not directory.is_dir() or not re.search(r'_(2425|2526)$', directory.name):
            continue
        league, code = directory.name.rsplit('_',1)
        year = 2000 + int(code[:2])
        for path in directory.glob('*.json'):
            key = f'{directory.name}/{path.name}'
            previous = stored.get(key)
            if previous and previous['model_hash'] == model_hash and previous['mtime'] == path.stat().st_mtime_ns:
                saved[key] = previous
                continue
            try:
                match = json.loads(path.read_text())
                day = date.fromisoformat(match['startDate'][:10])
                targets = []
                for side in ['home','away']:
                    for p in match[side]['players']:
                        t = BY_NAME.get(normalize(p['name']))
                        if not t or year != int(t['window'][-4:])-1 or day >= date.fromisoformat(t['transfer_date']):
                            continue
                        birth = births.get(t['player_id'])
                        # Historical WhoScored match responses expose current age, not match-date age.
                        snapshot = date(2026,11,4)
                        expected_age = snapshot.year - birth.year - ((snapshot.month,snapshot.day)<(birth.month,birth.day)) if birth else None
                        if expected_age is not None and p.get('age') is not None and abs(p['age']-expected_age)>1:
                            errors.append({'file':key,'player':t['player'],'error':'Name match rejected: age mismatch'})
                            continue
                        targets.append((side,p,t))
                rows = []
                if targets:
                    minutes = regulation_minutes(match)
                    entries = possession_entries(match['events'])
                    actions = spadl(path, league, year, match)
                    actions['xt'] = np.nan
                    coords = ['start_x','start_y','end_x','end_y']
                    usable = actions.dropna(subset=coords)
                    actions.loc[usable.index,'xt'] = model.rate(usable)
                    for side, p, t in targets:
                        pid = int(p['playerId'])
                        if minutes[pid] <= 0:
                            continue
                        own = actions[actions['player_id']==pid]['xt'].dropna()
                        shots = sum(bool(e.get('isShot')) and e.get('playerId')==pid
                            and (e.get('period') or {}).get('value') in [1,2]
                            and not any((q.get('type') or {}).get('displayName')=='OwnGoal' for q in e.get('qualifiers',[]))
                            for e in match['events'])
                        eligible = get_successful_move_actions(actions[actions['player_id']==pid])
                        missing_coordinates = int(eligible[coords].isna().any(axis=1).sum())
                        valid = trained_through < t['transfer_date'] and missing_coordinates == 0
                        rows.append({'player_id': t['player_id'], 'whoscored_player_id':pid, 'name':t['player'],
                            'league':league,'season_start':year,'game_id':int(path.stem),'date':day.isoformat(),
                            'team':match[side]['name'],'team_id':match[side]['teamId'],
                            'minutes':minutes[pid], 'shots':shots, 'receptions':entries[pid],
                            'xt_total':float(own.sum()) if valid else None,
                            'xt_positive':float(own[own>0].sum()) if valid else None,
                            'xt_model_trained_through':trained_through, 'xt_asof_valid':valid,
                            'xt_eligible_actions':len(eligible),'xt_missing_coordinate_actions':missing_coordinates,
                            'source_url':f'https://www.whoscored.com/Matches/{path.stem}/Live'})
                saved[key] = {'mtime':path.stat().st_mtime_ns, 'model_hash':model_hash, 'rows':rows}
            except Exception as exc:
                errors.append({'file':key,'error':str(exc)})
    cache_path.write_text(json.dumps(saved))
    matches = [r for s in saved.values() for r in s['rows']]
    fm = fotmob_stats()
    fm_matches = fotmob_match_stats(fm)
    fm_joint_index=defaultdict(list)
    for g in fm_matches:
        fm_joint_index[(g['player_id'],g['league'],g['date'],team_key(g['team']))].append(g)
    joint_matches=[]
    for g in matches:
        candidates=fm_joint_index[(g['player_id'],g['league'],g['date'],team_key(g['team']))]
        if len(candidates)==1:
            f=candidates[0]
            joint_matches.append({**g,'fotmob_game_id':f['game_id'],
                'xg':f['xg'],'npxg':f['npxg'],'fotmob_minutes':f['minutes']})
    dataset_status={}
    for f in (DATA/'collection').glob('*.json'):
        value=json.loads(f.read_text())
        dataset_status[(value['league'],value['season_start'])]=value
    episodes, injury_sources = injury_episodes()
    features, season_rows, coverage = [], [], []
    for t in POOL:
        pid = t['player_id']
        observed = [r for r in matches if r['player_id']==pid]
        seasons = [s for s in fm if s['player_id']==pid]
        requested = [int(t['window'][-4:])-1]
        coverage.append({'player_id':pid,'name':t['player'],'requested_seasons':'; '.join(f'{y}/{y+1}' for y in requested),
            'whoscored_seasons_observed':'; '.join(str(y) for y in sorted({r['season_start'] for r in observed})),
            'fotmob_seasons_observed':'; '.join(str(y) for y in sorted({s['season_start'] for s in seasons})),
            'whoscored_matches':len(observed),'injury_history_has_entries':bool(injury_sources.get(pid)),
            'fee_type':t['transfer_type']})
        if not observed:
            continue
        event_source_complete=bool(seasons) and all(
            dataset_status.get((s['league'],s['season_start']),{}).get('state')=='complete'
            and not dataset_status.get((s['league'],s['season_start']),{}).get('unmatched_team_names')
            for s in seasons)
        minute_reconciled=True
        for y in requested:
            reference_minutes=sum(s['xg_source_minutes'] for s in seasons if s['season_start']==y)
            event_minutes=sum(g['minutes'] for g in observed if g['season_start']==y)
            if not reference_minutes or abs(event_minutes-reference_minutes)>max(90,.05*reference_minutes):
                minute_reconciled=False
        expiry = t['old_contract_expiry']
        contract = max(0,(date.fromisoformat(expiry)-date.fromisoformat(t['transfer_date'])).days/365.25) if expiry else None
        inj, missed, denominator, inj_status = injury_percentage(pid,seasons,episodes,injury_sources,observed)
        all_years = [y for y in requested if any(r['season_start']==y for r in observed)]
        for year in all_years + [None]:
            games = [r for r in observed if year is None or r['season_start']==year]
            xg_rows = [s for s in seasons if year is None or s['season_start']==year]
            xg_games = [g for g in fm_matches if g['player_id']==pid and (year is None or g['season_start']==year)]
            xg_uncertainty=uncertainty([g['xg'] if g['xg'] is not None else np.nan for g in xg_games],
                                       [g['minutes'] for g in xg_games])
            row_inj,row_missed,row_denom,row_inj_status = injury_percentage(pid,xg_rows,episodes,injury_sources,games)
            total_minutes = sum(g['minutes'] for g in games)
            xt_valid = all(g['xt_asof_valid'] for g in games)
            xt = uncertainty([g['xt_total'] if g['xt_total'] is not None else np.nan for g in games],[g['minutes'] for g in games])
            xg_complete = bool(xg_rows) and all(s['xg_total'] is not None for s in xg_rows)
            xg_total = sum(s['xg_total'] for s in xg_rows) if xg_complete else None
            xg_minutes = sum(s['xg_source_minutes'] for s in xg_rows)
            match_xg_minutes=sum(g['minutes'] for g in xg_games)
            row = {'player_id':pid,'name':t['player'],'transfer_date':t['transfer_date'], 'window':t['window'],
                'season':f'{requested[0]}/{requested[0]+1}',
                'requested_prior_seasons':1,
                'seasons_with_event_data':len({g['season_start'] for g in games}),
                'total_minutes':total_minutes,'appearances_observed':len(games),
                'total_xt_added':sum(g['xt_total'] for g in games) if xt_valid else None,
                'total_positive_xt_added':sum(g['xt_positive'] for g in games) if xt_valid else None,
                'xt_per90':xt['rate_per90'],'total_receptions':sum(g['receptions'] for g in games),
                'reception_definition':'Observed on-ball actor transitions, including recoveries; event-derived proxy',
                'shots_per90':sum(g['shots'] for g in games)*90/total_minutes,
                'xg_per90':xg_total*90/xg_minutes if xg_total is not None and xg_minutes else None,
                'xg_total_fotmob':xg_total,'xg_minutes_fotmob':xg_minutes,
                'transfer_fee_eur':t['fee_eur'] if t['fee_eur']!='' else None,
                'transfer_type':t['transfer_type'],'selling_club':t['old_club'],'selling_club_league':t['old_club_league'],
                'positive_permanent_fee_eligible':t['transfer_type']!='Loan' and isinstance(t['fee_eur'],(int,float)) and t['fee_eur']>0,
                'injury_missed_league_matches_pct':row_inj,
                'injury_missed_league_matches':row_missed,
                'team_league_matches_denominator':row_denom,
                'injury_measure_status':row_inj_status,
                'years_until_selling_contract_expiry':contract,'selling_contract_expiry':expiry,
                'position':t['position'],'age_at_transfer':t['age_at_transfer'],
                'xt_weighted_match_sd_per90':xt['weighted_sd_per90'],'xt_se_per90':xt['se_per90'],
                'xt_effective_matches':xt['effective_matches'],
                'minutes_reliability_900':total_minutes/(total_minutes+900),
                'low_exposure_flag':total_minutes<900 or len(games)<10,
                'xt_one_se_lower_scenario':xt['rate_per90']-xt['se_per90'] if xt['se_per90'] is not None else None,
                'xt_model_trained_through':trained_through,'xt_asof_valid':xt_valid,
                'xg_source':'FotMob; published rounded totals; own source minutes denominator',
                'xg_seasons_observed':len({s['season_start'] for s in xg_rows}),
                'xg_matches_observed':len(xg_games),
                'xg_match_minutes_observed':sum(g['minutes'] for g in xg_games),
                'xg_match_minutes_reconciled':bool(xg_minutes and abs(match_xg_minutes-xg_minutes)<=max(90,.05*xg_minutes)),
                'xg_weighted_match_sd_per90':xg_uncertainty['weighted_sd_per90'],
                'xg_se_per90':xg_uncertainty['se_per90'],
                'xg_effective_matches':xg_uncertainty['effective_matches'],
                'paired_xg_xt_matches':sum(g['player_id']==pid and (year is None or g['season_start']==year) for g in joint_matches),
                'transfer_source_url':t['source_url'],
                'performance_coverage':'Provisional: reconcile expected team fixtures and minutes before modeling',
                'event_source_collection_complete':event_source_complete,
                'minutes_reconciled_to_season_source':minute_reconciled,
            }
            if year is None:
                yearly = []
                for y in requested:
                    ss = [s for s in xg_rows if s['season_start']==y]
                    if ss and all(s['xg_total'] is not None for s in ss):
                        yearly.append(sum(s['xg_total'] for s in ss)*90/sum(s['xg_source_minutes'] for s in ss))
                row['xg_sd_status'] = 'Match-level uncertainty within the one requested prior season'
                features.append(row)
            else:
                season_rows.append(row)
    write_csv(OUT/'player_transfer_features.csv',features)
    write_csv(OUT/'player_season_features.csv',season_rows)
    write_csv(OUT/'coverage_audit.csv',coverage)
    write_csv(OUT/'player_match_metrics.csv',matches)
    write_csv(OUT/'extraction_errors.csv',errors,fieldnames=['file','player','error'])
    visible_fm = [{k:v for k,v in s.items() if k!='team_games'} for s in fm]
    write_csv(OUT/'fotmob_player_league_seasons.csv',visible_fm)
    write_csv(OUT/'fotmob_player_match_xg.csv',fm_matches)
    write_csv(OUT/'paired_match_risk_inputs.csv',joint_matches)
    requested_columns=['total_minutes','total_xt_added','total_receptions','shots_per90','xg_per90',
        'transfer_fee_eur','injury_missed_league_matches_pct','years_until_selling_contract_expiry']
    for row in features:
        row['model_ready']=bool(row['positive_permanent_fee_eligible'] and row['seasons_with_event_data']==1
            and row['xg_seasons_observed']==1 and row['event_source_collection_complete']
            and row['minutes_reconciled_to_season_source'] and row['xg_match_minutes_reconciled']
            and all(row[k] is not None for k in requested_columns))
        if row['event_source_collection_complete'] and row['minutes_reconciled_to_season_source']:
            row['performance_coverage']='Requested event sources collected and minutes reconciled; inspect remaining field gaps'
    write_csv(OUT/'player_transfer_features.csv',features)
    write_csv(OUT/'model_ready_features.csv',[r for r in features if r['model_ready']],fieldnames=list(features[0]) if features else [])
    con = duckdb.connect(str(ROOT/'attacker_transfer_history.duckdb'))
    for table, records in [('attacker_transfer_metadata',POOL),('attacker_player_match_stats',matches),
        ('attacker_player_season_features',season_rows),('attacker_transfer_features',features),
        ('attacker_coverage_audit',coverage),('attacker_xg_source_seasons',visible_fm),
        ('attacker_xg_source_matches',fm_matches)]:
        if records:
            con.register('frame',pd.DataFrame(records))
            con.execute(f'CREATE OR REPLACE TABLE {table} AS SELECT * FROM frame')
            con.unregister('frame')
    if joint_matches:
        con.register('frame',pd.DataFrame(joint_matches))
        con.execute('CREATE OR REPLACE TABLE attacker_paired_match_risk_inputs AS SELECT * FROM frame')
        con.unregister('frame')
    con.close()
    plan=json.loads((DATA/'required_team_seasons.json').read_text())
    statuses=[dataset_status.get((key.rsplit('_',1)[0],int(key.rsplit('_',1)[1])),{}) for key in plan]
    fm_status_path=DATA/'fotmob/match_collection_status.json'
    fm_status=json.loads(fm_status_path.read_text()) if fm_status_path.exists() else {}
    summary = {'target_players':71,'players_with_observed_event_data':len(features),
        'requested_prior_seasons':1,
        'players_with_requested_prior_season_observed':sum(r['seasons_with_event_data']==1 for r in features),
        'observed_player_matches':len(matches),'player_season_rows':len(season_rows),
        'xg_source_player_league_seasons':len(fm),'processing_errors':len(errors),
        'xt_reference_trained_through':trained_through,
        'model_ready_players':sum(r['model_ready'] for r in features),
        'collection_finished':all(s.get('state') in ['complete','partial','failed'] for s in statuses) and fm_status.get('complete',False),
        'collection_complete':all(s.get('state')=='complete' and not s.get('unmatched_team_names') for s in statuses) and fm_status.get('complete',False)}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
