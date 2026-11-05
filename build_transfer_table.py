"""Build reviewable tables from browser-extracted Transfermarkt records."""
import csv
import html
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data/transfermarkt'
OUT = ROOT / 'outputs/transfermarkt'
OUT.mkdir(parents=True, exist_ok=True)

def iso(value):
    return datetime.strptime(value, '%d/%m/%Y').date().isoformat() if value else ''

def euros(value):
    if value == 'free transfer':
        return 0
    m = re.search(r'€([\d,.]+)([MK]?)', value)
    return round(float(m[1].replace(',', '')) * {'M': 1e6, 'K': 1e3, '': 1}[m[2]]) if m else ''

raw = json.loads((DATA / 'raw_summer_2025.json').read_text()) + [
    r for r in json.loads((DATA / 'raw_arrivals.json').read_text())
    if r['window'] == 'summer_2026'
]
details = json.loads((DATA / 'details_summer_2025.json').read_text()) + [
    r for r in json.loads((DATA / 'details_verified.json').read_text())
    if r['window'] == 'summer_2026'
]
assert len(raw) == 642 and len(details) == 339
assert {r['window'] for r in raw} == {'summer_2025', 'summer_2026'}
assert len({(r['window'], r['transfer_id']) for r in details}) == len(details)
rows = []
for r in details:
    fee = r['fee'].replace('\n', ' ')
    loan = 'loan' in fee.lower()
    same_club = r.get('current_club_id') == r['new_club_id']
    same_join = r.get('header_joined_date') == r.get('transfer_date')
    expiry = r.get('current_contract_expiry', '')
    match = same_club and same_join and bool(expiry)
    competitions = next((x.split('\t') for x in r['detail_rows'] if '\tCompetition\t' in x), [])
    old_league = competitions[0] if competitions else ''
    rows.append({
        'player': r['player'], 'old_club': r['old_club'], 'new_club': r['new_club'],
        'window': r['window'], 'transfer_date': iso(r.get('transfer_date')),
        'old_contract_expiry': iso(r.get('old_contract_expiry')),
        'old_contract_status': 'Reported on transfer detail page' if r.get('old_contract_expiry') else 'Not displayed; not inferred from fee',
        'new_club_contract_expiry_displayed': iso(expiry) if match else '',
        'new_contract_status': ('Current profile at receiving club; joined date matches transfer; original terms not independently verified' if match else
            'Current profile is a different club' if not same_club else
            'Current joined date differs; possible later loan/permanent move' if not same_join else 'Not displayed'),
        'fee_display': fee, 'fee_eur': euros(fee),
        'transfer_type': 'Loan' if loan else 'Free transfer' if fee == 'free transfer' else 'Permanent / unspecified',
        'age_at_transfer': int(r['age']), 'position': r['position'], 'nationality': r['nationality'],
        'market_value_at_transfer_display': r['market_value'], 'market_value_at_transfer_eur': euros(r['market_value']),
        'old_club_league': old_league, 'within_premier_league': old_league == 'Premier League',
        'profile_club_at_scrape': r.get('current_club', ''),
        'profile_contract_expiry_at_scrape': iso(expiry), 'profile_joined_date_at_scrape': iso(r.get('header_joined_date')),
        'player_id': r['player_id'], 'transfer_id': r['transfer_id'],
        'source_url': r['transfer_url'], 'profile_url': r['player_url'], 'retrieved_as_of': '2026-11-04',
    })
rows.sort(key=lambda r: (r['transfer_date'], r['new_club'], r['player']))
assert all(r['transfer_date'].startswith(r['window'][-4:] + '-') for r in rows)

def write_csv(path, records):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)

write_csv(OUT / 'premier_league_transfers.csv', rows)
(OUT / 'premier_league_transfers.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2))
signing_ids = {(r['window'], r['transfer_id']) for r in rows}
appendix = []
for r in raw:
    kind = 'Signing' if (r['window'], r['transfer_id']) in signing_ids else 'Loan return' if r['fee'].startswith('End of loan') else 'Internal promotion'
    appendix.append({**r, 'arrival_category': kind})
write_csv(OUT / 'all_arrival_entries.csv', appendix)
summary = {
    'as_of': '2026-11-04', 'windows': ['summer_2025', 'summer_2026'],
    'signing_transfers': len(rows), 'unique_signing_players': len({r['player_id'] for r in rows}),
    'transfers_by_window': dict(Counter(r['window'] for r in rows)),
    'old_contract_expiries_available': sum(bool(r['old_contract_expiry']) for r in rows),
    'receiving_club_profile_expiries_with_matching_join_date': sum(bool(r['new_club_contract_expiry_displayed']) for r in rows),
    'within_premier_league_transfers': sum(r['within_premier_league'] for r in rows),
    'all_arrival_entries': len(raw), 'unique_all_arrival_players': len({r['player_id'] for r in raw}),
    'arrival_categories': dict(Counter(r['arrival_category'] for r in appendix)),
}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2))

columns = [('player','Player'),('old_club','From'),('new_club','To'),('transfer_date','Date'),('old_contract_expiry','Old expiry'),('new_club_contract_expiry_displayed','New club expiry*'),('fee_display','Fee'),('transfer_type','Type'),('position','Position'),('age_at_transfer','Age'),('nationality','Nationality'),('market_value_at_transfer_display','Market value'),('old_club_league','From league'),('new_contract_status','Contract note')]
thead = ''.join('<th>'+html.escape(label)+'</th>' for _, label in columns)
tbody = ''
for r in rows:
    cells = []
    for key, _ in columns:
        value = html.escape(str(r[key]) or '—')
        if key == 'player': value = '<a target="_blank" rel="noopener" href="'+html.escape(r['source_url'])+'">'+value+'</a>'
        cells.append('<td>'+value+'</td>')
    tbody += '<tr>'+''.join(cells)+'</tr>\n'
headline = f"{summary['unique_signing_players']} unique players · {len(rows)} signing transfers"
window_counts = 'Summer 2025: 166 transfers; summer 2026: 173 transfers.'
excluded = (f"Excludes {summary['arrival_categories'].get('Loan return', 0)} loan-return entries and "
            f"{summary['arrival_categories'].get('Internal promotion', 0)} internal promotions, "
            'which are preserved in the complete arrivals CSV.')
page = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Premier League summer transfers · 2025 & 2026</title>
<style>body{font:14px system-ui;margin:28px;color:#172a38}h1{font-size:27px}p{max-width:1100px;line-height:1.5}.wrap{overflow:auto;max-height:73vh;border:1px solid #ccd5da}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:10px;border-bottom:1px solid #dde3e7;min-width:100px}th{position:sticky;top:0;background:#173c4e;color:white}tr:nth-child(even){background:#f2f6f8}a{color:#126282}input{padding:10px;width:350px;margin:10px 0}td:last-child{min-width:300px}</style>
<h1>Premier League arrivals · summer 2025 & summer 2026</h1>
<p><strong>'''+headline+'</strong> '+window_counts+''' Includes permanent signings, free transfers, incoming loans and moves between Premier League clubs. '''+excluded+'''</p>
<p><strong>*Contract dates:</strong> Old expiry comes from the individual transfer detail page. New club expiry is the current profile value only where both receiving club and joined date match this transfer. It may reflect a subsequent renewal; original new-contract terms are not independently verified. Blank means unavailable or a mismatched current profile, never zero. For loans, expiry may be the loan end, not the parent-club employment contract.</p>
<p>Fees and market values are Transfermarkt figures in EUR; unknown fees remain blank in the numeric CSV field. As of November 4, 2026. Click a player for the individual source. <a href="premier_league_transfers.csv">Download signing CSV</a> · <a href="all_arrival_entries.csv">All '''+str(len(raw))+''' arrival entries</a></p>
<input id="q" placeholder="Filter player, club, position or any field" aria-label="Filter transfers"><span id="count">'''+str(len(rows))+''' transfers</span>
<div class="wrap"><table><thead><tr>'''+thead+'</tr></thead><tbody>'+tbody+'''</tbody></table></div>
<script>document.getElementById('q').addEventListener('input',e=>{let n=0;for(const r of document.querySelectorAll('tbody tr')){r.hidden=!r.textContent.toLowerCase().includes(e.target.value.toLowerCase());if(!r.hidden)n++}document.getElementById('count').textContent=n+' transfers'})</script></html>'''
(OUT / 'transfers.html').write_text(page)
print(json.dumps(summary, indent=2))
