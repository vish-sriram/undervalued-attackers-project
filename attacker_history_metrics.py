"""Auditable event-derived metrics and uncertainty for the attacker cohort."""
from collections import Counter
import math
import unicodedata
import numpy as np

NAME_ALIASES = {
    'bengannondoak': 'bendoak', 'jamiebynoegittens': 'jamiegittens',
    'savinho': 'savio', 'strandlarsen': 'jorgenstrandlarsen',
    'jeanmatteobahoya': 'jeanmatteobahoya',
    'tommywatson': 'tomwatson', 'arnaudkalimuendomuinga': 'arnaudkalimuendo',
    'yeremipino': 'yeremypino', 'emanuelemegha': 'emmanuelemegha',
}

def normalize(name):
    value = ''.join(c for c in unicodedata.normalize('NFKD', name.lower()) if c.isalnum())
    return NAME_ALIASES.get(value, value)

TEAM_ALIASES = {
    'manchestercity': 'mancity', 'manchesterunited': 'manutd',
    'tottenhamhotspur': 'tottenham', 'brightonhovealbion': 'brighton',
    'wolverhamptonwanderers': 'wolves', 'nottinghamforest': 'forest',
    'parissaintgermain': 'psg', 'bayer04leverkusen': 'leverkusen',
    'bayerleverkusen': 'leverkusen', 'eintrachtfrankfurt': 'frankfurt',
    'bayernmunchen': 'bayern', 'fcbayernmunich': 'bayern',
    'ajaxamsterdam': 'ajax', 'ogcnice': 'nice', 'rcstrasbourgalsace': 'strasbourg',
    'losclille': 'lille', 'asroma': 'roma', 'acmilan': 'milan',
    'staderennaisfc': 'rennes', 'fctoulouse': 'toulouse', 'vflwolfsburg': 'wolfsburg',
    'tsg1899hoffenheim': 'hoffenheim', '1fcunionberlin': 'unionberlin',
    'estactroyes': 'troyes', 'celta': 'celtavigo', 'celtadevigo': 'celtavigo',
    'realbetisbalompie': 'realbetis', 'caosasuna': 'osasuna',
    'athleticclub': 'athleticbilbao', 'rb': 'rbleipzig',
    'brightonandhovealbion':'brighton', 'ipswichtown':'ipswich',
    'newcastleunited':'newcastle', 'westhamunited':'westham',
    'staderennais':'rennes', 'borussiadortmund':'dortmund',
    'stuttgart':'vfbstuttgart',
}

def team_key(name):
    name = normalize(name)
    if name.endswith('fc'):
        name = name[:-2]
    if name.startswith('afc'):
        name = name[3:]
    return TEAM_ALIASES.get(name, name)

def possession_entries(events):
    """Count observed changes of on-ball actor, not touches or inferred carries.

    This is an event-sequence proxy for the user's possession-change definition.
    It includes recoveries from opponents. It is not a verified pass-recipient feed.
    """
    counts = Counter()
    holder = None
    period = None
    for event in events:
        p = (event.get('period') or {}).get('value')
        if p not in (1, 2):
            continue
        if p != period:
            holder, period = None, p
        kind = (event.get('type') or {}).get('displayName')
        if kind in {'End', 'OffsideGiven', 'CornerAwarded', 'Goal'}:
            # A goal is counted as an on-ball event below, then ends possession.
            if kind != 'Goal':
                holder = None
                continue
        pid = event.get('playerId')
        if pid is None or not event.get('isTouch', False):
            continue
        actor = (int(event.get('teamId', -1)), int(pid))
        if actor != holder:
            counts[int(pid)] += 1
        holder = None if kind == 'Goal' else actor
    return counts

def regulation_minutes(match):
    """Approximate conventional league minutes; exclude added-time inflation."""
    state = {}
    for side in ['home', 'away']:
        for p in match[side]['players']:
            state[int(p['playerId'])] = {'on': 0.0 if p.get('isFirstEleven') else None, 'minutes': 0.0}
    for e in match['events']:
        pid = e.get('playerId')
        period = (e.get('period') or {}).get('value')
        if pid is None or int(pid) not in state or period not in (1, 2):
            continue
        minute = float(e.get('minute', 0)) + float(e.get('second', 0))/60
        minute = min(45 if period == 1 else 90, minute)
        kind = (e.get('type') or {}).get('displayName')
        labels = {(q.get('type') or {}).get('displayName') for q in e.get('qualifiers', [])}
        off = kind == 'SubstitutionOff' or (kind == 'Card' and bool(labels & {'Red', 'SecondYellow'}))
        s = state[int(pid)]
        if off and s['on'] is not None:
            s['minutes'] += max(0.0, minute - s['on'])
            s['on'] = None
        elif kind == 'SubstitutionOn' and s['on'] is None:
            s['on'] = minute
    for s in state.values():
        if s['on'] is not None:
            s['minutes'] += max(0.0, 90.0 - s['on'])
    return {pid: s['minutes'] for pid, s in state.items()}

def uncertainty(totals, minutes):
    """Match-weighted dispersion and a cluster-by-match SE of a per-90 ratio.

    SE is uncertainty of the average; SD describes match-to-match variation.
    They are deliberately separate. No euro penalty is invented here.
    """
    x, m = np.asarray(totals, dtype=float), np.asarray(minutes, dtype=float)
    good = np.isfinite(x) & np.isfinite(m) & (m > 0)
    x, m = x[good], m[good]
    if not len(x):
        return {'rate_per90': None, 'weighted_sd_per90': None, 'se_per90': None, 'effective_matches': 0.0}
    w, rates = m/90.0, x*90.0/m
    mean = float(x.sum()/w.sum())
    neff = float(w.sum()**2 / (w*w).sum())
    denom = w.sum() - (w*w).sum()/w.sum()
    sd = math.sqrt(float((w*(rates-mean)**2).sum()/denom)) if denom > 0 else None
    residual = x - mean*w
    se = math.sqrt(float(len(x)/(len(x)-1)*(residual*residual).sum()))/w.sum() if len(x) >= 5 and m.sum() >= 180 else None
    return {'rate_per90': mean, 'weighted_sd_per90': sd, 'se_per90': se, 'effective_matches': neff}

def risk_adjusted_rate(rate, minutes, prior_rate, se, match_sd, *, prior_minutes=900,
                       uncertainty_weight=1.0, volatility_weight=0.0):
    """Configurable scenario, not a fitted or validated financial valuation.

    Estimate prior_rate inside the training fold, by role. Tune the exposure
    prior and penalty weights only on training/validation data. Missing SE must
    remain missing rather than giving a low-exposure player zero risk.
    """
    reliability = minutes / (minutes + prior_minutes)
    center = reliability*rate + (1-reliability)*prior_rate
    if se is None or (volatility_weight and match_sd is None):
        return {'reliability':reliability, 'shrunken_rate':center, 'risk_adjusted_rate':None}
    penalty = uncertainty_weight*se + volatility_weight*(match_sd or 0)
    return {'reliability':reliability,'shrunken_rate':center,
            'risk_adjusted_rate':center-penalty}
