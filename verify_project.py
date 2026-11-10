"""Verify published fee-model and shortlist outputs without network access."""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from fit_fee_arrivals_only import FEATURES, LEAGUES, OUT, fit
from predict_transfer_fee import predict_fees


def verify():
    deals = pd.read_csv(OUT / "model_deals.csv")
    train = deals[deals.window.eq("summer_2025")]
    test = deals[deals.window.eq("summer_2026")]
    assert (len(train), len(test)) == (19, 14)
    assert deals.transfer_id.is_unique and deals.player_id.is_unique
    assert deals.buying_club_league.eq("Premier League").all()
    assert (deals.transfer_fee_eur > 0).all()
    assert set(train.player_id).isdisjoint(test.player_id)
    np.testing.assert_allclose(
        np.log(deals.prior_season_minutes), deals.ln_prior_season_minutes
    )

    saved = pd.read_csv(OUT / "holdout_predictions.csv").set_index("transfer_id")
    assert set(saved.index) == set(test.transfer_id)
    expected = test.transfer_id.map(saved.prediction_eur_m)
    actual = test.transfer_fee_eur / 1e6
    np.testing.assert_allclose(
        test.transfer_fee_eur, test.transfer_id.map(saved.transfer_fee_eur)
    )
    rebuilt, alpha = fit(train)
    np.testing.assert_allclose(
        rebuilt.predict(test[FEATURES + LEAGUES]), expected, rtol=1e-10
    )
    np.testing.assert_allclose(
        predict_fees(test).predicted_fee_eur, expected * 1e6, rtol=0, atol=0.005001
    )
    report = json.loads((OUT / "metrics.json").read_text())
    assert alpha == report["alpha"] == 10
    scores = {
        "mae_eur_m": mean_absolute_error(actual, expected),
        "rmse_eur_m": np.sqrt(np.mean((actual - expected) ** 2)),
        "r_squared_euro": r2_score(actual, expected),
        "r_squared_log": r2_score(np.log(actual), np.log(expected)),
    }
    for key, value in scores.items():
        np.testing.assert_allclose(value, report["arrivals_only"][key])

    shortlist = OUT.parent.parent / "attacker_valuations_2026"
    players = pd.read_csv(shortlist / "player_valuations.csv")
    assert len(players) == players.whoscored_player_id.nunique() == 34
    priced = players[players.predicted_fee_eur.notna()]
    assert len(priced) == 32
    np.testing.assert_allclose(
        predict_fees(priced).predicted_fee_eur, priced.predicted_fee_eur
    )
    filtered = pd.read_csv(shortlist / "filtered_player_valuations.csv")
    assert len(filtered) == 7 and filtered.predicted_fee_eur.notna().sum() == 6
    assert not filtered.selling_club.isin(
        ["Arsenal", "Manchester City", "Liverpool"]
    ).any()
    assert set(filtered.whoscored_player_id) <= set(players.whoscored_player_id)
    np.testing.assert_allclose(
        filtered.predicted_fee_eur,
        filtered.whoscored_player_id.map(
            players.set_index("whoscored_player_id").predicted_fee_eur
        ),
        equal_nan=True,
    )
    return scores
