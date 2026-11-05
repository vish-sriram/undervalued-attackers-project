"""Audit cached action scores and finish the interrupted shortlist report."""
from collections import Counter
from datetime import date
import gzip
import json

import numpy as np
import pandas as pd

from build_attacker_history import reference_model
from predict_transfer_fee import predict_fees, NUMERIC_INPUTS
from score_attacker_shortlist import OUT, build_inputs, player_sources


def main():
    pool = pd.read_csv(OUT / 'screened_players.csv')
    actions = pd.read_csv(OUT / 'player_actions_xt.csv.gz')
    games = pd.read_csv(OUT / 'player_match_minutes.csv')
    counts = Counter()
    with gzip.open(OUT / 'player_source_events.jsonl.gz', 'rt') as stream:
        for line in stream:
            counts[int(json.loads(line)['event']['playerId'])] += 1
    assert pool.pid.nunique() == len(pool) == 34
    assert not actions.duplicated(['league', 'game_id', 'action_id']).any()
    assert not games.duplicated(['player_id', 'league', 'game_id']).any()
    model, cutoff, _ = reference_model()
    scored = actions[actions.xt_added.notna()]
    np.testing.assert_allclose(model.rate(scored), scored.xt_added, atol=1e-12)
    assert scored.xt_eligible.all()
    assert actions.match_date.min() > cutoff
    rows = build_inputs(pool, player_sources(pool), games, actions, counts)
    overrides = json.loads((OUT / 'source_overrides.json').read_text())
    rows['source_notes'] = ''
    rows['additional_source_urls'] = ''
    for pid, values in overrides.items():
        mask = rows.whoscored_player_id.eq(int(pid))
        assert mask.sum() == 1
        for key in ['selling_club', 'selling_club_league', 'selling_contract_expiry']:
            rows.loc[mask, key] = values[key]
        valuation_date = date.fromisoformat(rows.loc[mask, 'valuation_date'].iloc[0])
        rows.loc[mask, 'years_until_selling_contract_expiry'] = (
            date.fromisoformat(values['selling_contract_expiry']) - valuation_date).days / 365.25
        issues = rows.loc[mask, 'validation_status'].iloc[0].split('; ')
        issues = [x for x in issues if x not in {'Current selling club or league unverified',
                                               'Current contract expiry unavailable or expired'}]
        rows.loc[mask, 'validation_status'] = '; '.join(issues) if issues else 'validated'
        rows.loc[mask, 'source_notes'] = values['note']
        rows.loc[mask, 'additional_source_urls'] = ' | '.join(values['sources'])
    ready = rows.validation_status.eq('validated')
    predictions = predict_fees(rows[ready])
    rows = rows.merge(predictions[['whoscored_player_id', 'predicted_fee_eur']], how='left',
                      on='whoscored_player_id', validate='one_to_one')
    rows['validation_status'] = rows.validation_status.replace('validated', 'input_checks_passed')
    root = OUT.parent / 'fee_model_international'
    train = pd.read_csv(root / 'deals_and_audit.csv')
    train = train[train.exclusion_reason.isna() & train.xt_per90.notna()
                  & train.buying_league_premier_league.eq(1) & train.window.eq('summer_2025')]
    train = train.merge(pd.read_csv(OUT.parent / 'fee_model_xgboost/model_deals.csv')[
        ['transfer_id', 'total_xt_added']], on='transfer_id', validate='one_to_one')
    train['total_xg'] = train.xg_per90 * train.prior_season_minutes / 90
    assert len(train) == 19
    rows['outside_training_range'] = rows.apply(lambda r: '; '.join(
        c for c in NUMERIC_INPUTS if pd.notna(r[c]) and not train[c].min() <= r[c] <= train[c].max()), axis=1)
    rows['seller_league_unseen_in_training'] = ~rows.selling_club_league.isin(train.selling_club_league)
    holdout = pd.read_csv(root / 'arrivals_only/holdout_predictions.csv')
    np.testing.assert_allclose(predict_fees(holdout).predicted_fee_eur,
                               holdout.prediction_eur_m * 1e6, atol=.005001, rtol=0)
    rows = rows.sort_values('predicted_fee_eur', ascending=False, na_position='last')
    rows.to_csv(OUT / 'player_valuations.csv', index=False)
    summary = dict(players=len(rows), priced=int(ready.sum()), withheld=int((~ready).sum()),
                   raw_events=sum(counts.values()), actions=len(actions), scored_actions=len(scored),
                   eligible_missing_coordinates=int((actions.xt_eligible & actions.xt_added.isna()).sum()),
                   valuation_date=rows.valuation_date.iloc[0], reviewed_on='2026-11-05',
                   checks=['unique actions and player matches', 'all non-null xT scores recomputed',
                           'raw event counts reconciled', 'all 14 saved holdout predictions reproduced'],
                   limitation='Internal consistency and source corroboration, not independent video accuracy.')
    (OUT / 'audit.json').write_text(json.dumps(summary, indent=2) + '\n')
    text = ['# Attacker model estimates', '',
            'Valuation snapshot: 5 November 2026. Hypothetical permanent move to a Premier League buyer. For loans, the selling club is the parent club; loan options are not priced.', '',
            'The retained model exponentiates predicted log fees. These are model point estimates, not calibrated arithmetic expected values, bids, or independently verified market values. No retraining or retransformation correction was applied.', '',
            '19 training deals; 14 holdout deals; holdout euro R² 0.697 and MAE €13.79m. Performance on this small holdout does not establish accuracy for these players. Current injury, loan options, and release clauses are not model inputs.', '',
            f"{summary['raw_events']:,} raw events; {len(actions):,} converted actions; {len(scored):,} scored successful move actions. Shots and other ineligible actions have blank xT, not zero. Amad has one eligible action with missing endpoint coordinates; no endpoint was invented.", '',
            'The 34-player screen uses the available 2025/26 cache, with partial French and German league coverage. Player minutes are cross-checked against FotMob within max(20 minutes, 2%). Identities use league/name matching and age checks, not independent proof for every player. Raw events and converted actions are separate files because the conversion can infer carries and omit non-action events.', '',
            '| Player | Model fee (€m) | Status / extrapolation |', '|---|---:|---|']
    for r in rows.itertuples():
        fee = f'{r.predicted_fee_eur / 1e6:.2f}' if pd.notna(r.predicted_fee_eur) else '—'
        notes = r.validation_status
        if r.outside_training_range:
            notes += '; outside training range: ' + r.outside_training_range
        if r.seller_league_unseen_in_training:
            notes += '; seller league unseen in training'
        text.append(f'| {r.name} | {fee} | {notes} |')
    text.extend(['', 'Source-specific corrections and URLs are retained in source_overrides.json and the valuation table. Woltemade’s missing Stuttgart xG is resolved from a match shot map with no shots attributed to him, as recorded in woltemade_stuttgart_xg_verification.json. Ilić remains unpriced: his club confirms an extension without an expiry date, and third-party reports do not securely establish the new term.'])
    (OUT / 'README.md').write_text('\n'.join(text) + '\n')
    print(json.dumps(summary, indent=2))
    print(rows[['name', 'predicted_fee_eur', 'validation_status']].to_string(index=False))


if __name__ == '__main__':
    main()
