"""Supervise current downloads, reconcile gaps, rebuild and verify outputs.

This process continues the authorized collection after the interactive turn.
Progress and errors are persisted; completion is never inferred from launching.
"""
import concurrent.futures
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data/player_history'
OUT=ROOT/'outputs/attacker_history'
PYTHON=str(ROOT/'.venv/bin/python')
TERMINAL={'complete','partial','failed'}

def state(value):
    path=OUT/'run_status.json'
    value['updated_at']=datetime.now(timezone.utc).isoformat()
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def run(script,*args):
    with (DATA/f'{Path(script).stem}_supervised.log').open('a') as log:
        subprocess.run([PYTHON,str(ROOT/script),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)

def reconcile_events():
    # Own the current one-season job; no dependency on the cancelled two-season run.
    run('collect_attacker_history.py','--workers','3')

def reconcile_xg():
    run('collect_fotmob_matches.py')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    state({'state':'collecting','requested_prior_seasons':1,'message':'One prior season per transfer: 2024/25 for summer 2025; 2025/26 for summer 2026. Outputs are provisional.'})
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures=[executor.submit(reconcile_events),executor.submit(reconcile_xg)]
            last_build=0
            while not all(f.done() for f in futures):
                for f in futures:
                    if f.done():
                        f.result()
                if time.monotonic()-last_build>300:
                    run('build_attacker_history.py')
                    last_build=time.monotonic()
                    summary=json.loads((OUT/'summary.json').read_text())
                    state({'state':'collecting','summary':summary})
                    print(json.dumps(summary),flush=True)
                time.sleep(30)
            for f in futures:
                f.result()
        run('build_attacker_history.py')
        run('verify_attacker_history.py')
        summary=json.loads((OUT/'summary.json').read_text())
        state({'state':'finished' if summary['collection_complete'] else 'finished_with_source_gaps',
               'summary':summary,'message':'Review coverage_audit.csv and model_ready_features.csv before fitting.'})
        print(json.dumps(summary),flush=True)
    except Exception as exc:
        state({'state':'needs_attention','error_type':type(exc).__name__,'error':str(exc)})
        raise

if __name__=='__main__':
    main()
