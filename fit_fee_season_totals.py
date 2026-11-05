"""Fit the log-ridge transfer fee model using prior-season xG and xT totals."""
import json
from datetime import date

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from compare_log_fee_models import estimator
from fit_fee_hedonic import metrics
from fit_fee_international import OUTPUT, appearances
from fit_fee_ridge_outgoings import BUYER, league_stats, player_birth
from fit_fee_xgboost import LEAGUES, NUMERIC, OUTPUT as XT_OUTPUT, outgoing_xt

FEATURES = ["total_xg" if c == "xg_per90" else "total_xt_added" if c == "xt_per90" else c
            for c in NUMERIC] + ["is_winger", "tournament_appearances"]
AS_OF = "2026-11-04"
DOUE_ID = 914562
DOUE_FOTMOB_ID = 1339208


def fit(train):
    model = estimator(make_pipeline(StandardScaler(), Ridge()), FEATURES)
    search = GridSearchCV(model, {"regressor__model__ridge__alpha": [0.1, 1, 10, 100]},
                          cv=LeaveOneOut(), scoring="neg_mean_absolute_error")
    search.fit(train[FEATURES + LEAGUES], train.transfer_fee_eur / 1e6)
    return search.best_estimator_, search.best_params_["regressor__model__ridge__alpha"]


def doue_inputs():
    birth = player_birth(DOUE_FOTMOB_ID)
    assert birth == date(2005, 6, 3)
    season = [r for r in league_stats() if r[0] == 2025 and r[3] == DOUE_FOTMOB_ID]
    assert len(season) == 1 and season[0][1] == "FRA-Ligue 1"
    minutes, xg = season[0][4:]
    appearances_by_tournament = appearances(DOUE_FOTMOB_ID, AS_OF)
    data = pd.DataFrame([{
        "name": "Désiré Doué", "player_id": DOUE_ID, "transfer_id": -1,
        "season_start": 2025, "transfer_date": AS_OF, "prior_season_minutes": minutes,
        "ln_prior_season_minutes": np.log(minutes), "total_xg": xg, "age_at_transfer": 21,
        "years_until_selling_contract_expiry": (date(2029, 6, 30) - date.fromisoformat(AS_OF)).days / 365.25,
        "is_winger": 1, "tournament_appearances": sum(appearances_by_tournament.values()),
        "selling_club_league": "Ligue 1", "buying_club_league": "Premier League",
        **appearances_by_tournament,
    }])
    xt = outgoing_xt(data)
    if len(xt) != 1:
        raise ValueError("Doué's WhoScored season did not pass the existing xT coverage checks")
    return data.merge(xt, on="transfer_id", validate="one_to_one")


def main():
    audit = pd.read_csv(OUTPUT / "deals_and_audit.csv")
    data = audit[audit.exclusion_reason.isna() & audit.xt_per90.notna()].copy()
    totals = pd.read_csv(XT_OUTPUT / "model_deals.csv")[["transfer_id", "total_xt_added", "xt_event_minutes"]]
    data = data.merge(totals, on="transfer_id", validate="one_to_one")
    data["total_xg"] = (data.xg_per90 * data.prior_season_minutes / 90).round(8)
    train = data[data.window.eq("summer_2025")]
    test = data[data.window.eq("summer_2026")].copy()
    assert (len(train), len(test)) == (27, 25)
    assert set(train.player_id).isdisjoint(test.player_id)
    assert np.isfinite(data[FEATURES]).all().all()
    model, alpha = fit(train)
    test["prediction_eur_m"] = model.predict(test[FEATURES + LEAGUES])
    inbound = test[test[BUYER].eq(1)]
    summary = {"target": "natural log of fee in EUR millions", "alpha": alpha,
               "train_n": len(train), "test_n": len(test),
               "holdout_all": metrics(test.transfer_fee_eur / 1e6, test.prediction_eur_m),
               "holdout_arrivals": metrics(inbound.transfer_fee_eur / 1e6, inbound.prediction_eur_m)}
    encoded = model.regressor_["features"]
    pipeline = model.regressor_["model"]
    scaler, ridge = pipeline["standardscaler"], pipeline["ridge"]
    beta = ridge.coef_ / scaler.scale_
    intercept = ridge.intercept_ - scaler.mean_ @ beta
    summary["coefficients"] = dict(zip(["intercept", *encoded.get_feature_names_out()], [intercept, *beta]))
    np.testing.assert_allclose(np.exp(intercept + encoded.transform(test[FEATURES + LEAGUES]) @ beta),
                               test.prediction_eur_m)
    isak = train[train.name.eq("Alexander Isak")].copy()
    assert len(isak) == 1
    isak["prediction_eur_m"] = model.predict(isak[FEATURES + LEAGUES])
    leave_out_model, leave_out_alpha = fit(train[~train.player_id.isin(isak.player_id)])
    isak["prediction_excluding_own_deal_eur_m"] = leave_out_model.predict(isak[FEATURES + LEAGUES])
    summary["isak_leave_out_alpha"] = leave_out_alpha
    dump(model, OUTPUT / "season_totals_ridge.joblib")
    restored = load(OUTPUT / "season_totals_ridge.joblib")
    np.testing.assert_allclose(restored.predict(test[FEATURES + LEAGUES]), test.prediction_eur_m)
    test.to_csv(OUTPUT / "season_totals_holdout.csv", index=False)
    (OUTPUT / "season_totals_metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    cols = ["name", "prior_season_minutes", "xt_event_minutes", "total_xg", "total_xt_added",
            "years_until_selling_contract_expiry", "tournament_appearances", "prediction_eur_m",
            "prediction_excluding_own_deal_eur_m", "transfer_fee_eur"]
    doue = doue_inputs()
    doue["prediction_eur_m"] = model.predict(doue[FEATURES + LEAGUES])
    estimates = pd.concat([isak, doue], ignore_index=True)
    estimates[cols + ["selling_club_league", "buying_club_league", "transfer_date", "season_start",
                      "world_cup_appearances", "euro_appearances", "copa_america_appearances"]].to_csv(
        OUTPUT / "season_totals_player_estimates.csv", index=False)
    print(estimates[cols].to_string(index=False))


if __name__ == "__main__":
    main()
