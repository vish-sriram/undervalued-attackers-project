"""Refit the season-totals fee model on deals with a Premier League buyer."""
import json

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fit_fee_hedonic import metrics
from predict_transfer_fee import MODEL

OUT = MODEL.parent
OUTPUT = OUT.parent
FEATURES = ["age_at_transfer", "years_until_selling_contract_expiry", "ln_prior_season_minutes",
            "total_xg", "total_xt_added", "is_winger", "tournament_appearances"]
LEAGUES = ["selling_club_league", "buying_club_league"]


def fit(train):
    encoding = ColumnTransformer([
        ("numeric", "passthrough", FEATURES),
        ("league", OneHotEncoder(drop="first", min_frequency=3, handle_unknown="infrequent_if_exist",
                                 sparse_output=False), LEAGUES),
    ])
    regressor = Pipeline([("features", encoding), ("model", make_pipeline(StandardScaler(), Ridge()))])
    model = TransformedTargetRegressor(regressor=regressor, func=np.log, inverse_func=np.exp)
    search = GridSearchCV(model, {"regressor__model__ridge__alpha": [0.1, 1, 10, 100]},
                          cv=LeaveOneOut(), scoring="neg_mean_absolute_error")
    search.fit(train[FEATURES + LEAGUES], train.transfer_fee_eur / 1e6)
    return search.best_estimator_, search.best_params_["regressor__model__ridge__alpha"]


def main():
    OUT.mkdir(exist_ok=True)
    data = pd.read_csv(OUT / "model_deals.csv")
    train = data[data.window.eq("summer_2025")]
    test = data[data.window.eq("summer_2026")].copy()
    assert (len(train), len(test)) == (19, 14)
    assert set(train.player_id).isdisjoint(test.player_id)
    assert data.transfer_id.is_unique and data.buying_club_league.eq("Premier League").all()
    features = FEATURES + LEAGUES
    model, alpha = fit(train)
    test["prediction_eur_m"] = model.predict(test[features])
    pooled = pd.read_csv(OUTPUT / "season_totals_holdout.csv").set_index("transfer_id")
    test["pooled_prediction_eur_m"] = test.transfer_id.map(pooled.prediction_eur_m)
    actual = test.transfer_fee_eur / 1e6
    baseline = np.full(len(test), train.transfer_fee_eur.median() / 1e6)
    summary = {"training_deals": len(train), "test_deals": len(test), "alpha": alpha,
               "target": "natural log of transfer fee in EUR millions",
               "arrivals_only": metrics(actual, test.prediction_eur_m),
               "pooled_comparison": metrics(actual, test.pooled_prediction_eur_m),
               "training_median_baseline": metrics(actual, baseline)}
    test.to_csv(OUT / "holdout_predictions.csv", index=False)
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    isak = train[train.name.eq("Alexander Isak")].copy()
    isak["prediction_eur_m"] = model.predict(isak[features])
    leave_out, _ = fit(train[~train.player_id.isin(isak.player_id)])
    isak["prediction_excluding_own_deal_eur_m"] = leave_out.predict(isak[features])
    doue = pd.read_csv(OUTPUT / "season_totals_player_estimates.csv").query('name == "Désiré Doué"').copy()
    doue["age_at_transfer"] = 21
    doue["is_winger"] = 1
    doue["ln_prior_season_minutes"] = np.log(doue.prior_season_minutes)
    doue["prediction_eur_m"] = model.predict(doue[features])
    players = pd.concat([test[test.name.eq("Bradley Barcola")], isak, doue], ignore_index=True)
    cols = ["name", "prediction_eur_m", "prediction_excluding_own_deal_eur_m", "transfer_fee_eur"]
    players[cols + features].to_csv(OUT / "player_estimates.csv", index=False)
    path = OUT / "season_totals_ridge.joblib"
    dump(model, path)
    np.testing.assert_allclose(load(path).predict(test[features]), test.prediction_eur_m)
    print(json.dumps(summary, indent=2))
    print(players[cols].to_string(index=False))


if __name__ == "__main__":
    main()
