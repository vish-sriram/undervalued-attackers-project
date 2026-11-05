"""Build the xT-covered fee sample and evaluate a time-split XGBoost model."""
from collections import defaultdict
import json

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from build_attacker_history import CACHE, normalize, regulation_minutes, reference_model, spadl, get_successful_move_actions
from fit_fee_hedonic import metrics
from fit_fee_ridge_outgoings import BASE, BUYER, OUT, ROOT

OUTPUT = ROOT / "outputs/fee_model_xgboost"
COORDS = ["start_x", "start_y", "end_x", "end_y"]
LEAGUES = ["selling_club_league", "buying_club_league"]
NUMERIC = BASE + ["xt_per90"]
ALIASES = {"sonheungmin": "heungminson", "datrofofana": "daviddatrofofana"}


def outgoing_xt(data):
    targets = {(r.season_start, normalize(r.name)): r for r in data.itertuples()}
    games = defaultdict(list)
    for folder in sorted(CACHE.iterdir()):
        if not folder.is_dir() or not folder.name.endswith(("_2425", "_2526")):
            continue
        league, code = folder.name.rsplit("_", 1)
        year = int("20" + code[:2])
        for path in sorted(folder.glob("*.json")):
            match = json.loads(path.read_text())
            if not match or not match.get("events"):
                continue
            found = []
            for side in ("home", "away"):
                for player in match[side]["players"]:
                    name = normalize(player["name"])
                    target = targets.get((year, ALIASES.get(name, name)))
                    if target and match["startDate"][:10] < target.transfer_date:
                        found.append((target.transfer_id, int(player["playerId"])))
            if found:
                minutes = regulation_minutes(match)
                for deal, pid in found:
                    if minutes[pid] > 0:
                        games[deal].append((path, league, year, pid, minutes[pid]))

    model, cutoff, _ = reference_model()
    eligible = {}
    for deal in data.itertuples():
        rows = games[deal.transfer_id]
        minutes = sum(row[4] for row in rows)
        if (rows and len({row[3] for row in rows}) == 1 and cutoff < deal.transfer_date
                and abs(minutes - deal.prior_season_minutes) <= max(90, 0.05 * deal.prior_season_minutes)):
            eligible[deal.transfer_id] = rows
    by_match = defaultdict(list)
    for deal, rows in eligible.items():
        for path, league, year, pid, minutes in rows:
            by_match[path, league, year].append((deal, pid, minutes))

    print(f"Computing xT across {len(by_match)} cached matches for {len(eligible)} departures", flush=True)
    totals, exposure = defaultdict(float), defaultdict(float)
    invalid = set()
    for (path, league, year), players in by_match.items():
        match = json.loads(path.read_text())
        actions = spadl(path, league, year, match)
        for deal, pid, minutes in players:
            moves = get_successful_move_actions(actions.loc[actions.player_id.eq(pid)])
            if moves[COORDS].isna().any().any():
                invalid.add(deal)
                continue
            values = model.rate(moves)
            if not np.isfinite(values).all():
                invalid.add(deal)
                continue
            totals[deal] += values.sum()
            exposure[deal] += minutes
    return pd.DataFrame([
        {"transfer_id": deal, "xt_event_minutes": exposure[deal], "total_xt_added": total,
         "xt_per90": total * 90 / exposure[deal]}
        for deal, total in totals.items() if deal not in invalid
    ])


def main():
    OUTPUT.mkdir(exist_ok=True)
    data = pd.read_csv(OUT / "model_deals.csv")
    inbound = pd.read_csv(ROOT / "outputs/attacker_history/model_ready_features.csv")
    inbound["transfer_id"] = inbound.transfer_source_url.str.rsplit("/", n=1).str[-1].astype(int)
    xt_columns = ["transfer_id", "total_xt_added", "xt_per90", "xt_event_minutes"]
    existing = inbound.rename(columns={"total_minutes": "xt_event_minutes"})[xt_columns]
    extracted = outgoing_xt(data.loc[data[BUYER].eq(0)])
    data = data.merge(pd.concat([existing, extracted]), on="transfer_id", validate="one_to_one")
    assert np.isfinite(data[NUMERIC]).all().all() and data[LEAGUES].notna().all().all()
    assert data.player_id.is_unique and (data.transfer_fee_eur > 0).all()
    np.testing.assert_allclose(data.xt_per90, 90 * data.total_xt_added / data.xt_event_minutes)
    train = data[data.window.eq("summer_2025")]
    test = data[data.window.eq("summer_2026")].copy()
    assert set(train.player_id).isdisjoint(test.player_id)
    xgb = XGBRegressor(tree_method="hist", learning_rate=0.05, min_child_weight=3,
                       reg_lambda=5, random_state=2026, n_jobs=1)
    pipelines = {
        "xgboost": (xgb, {"regressor__model__max_depth": [1, 2], "regressor__model__n_estimators": [25, 60, 120]}),
        "ridge": (make_pipeline(StandardScaler(), Ridge()), {"regressor__model__ridge__alpha": [0.1, 1, 10, 100]}),
    }
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    summary = {"target": "natural log of fee in millions of euros",
               "train_deals": len(train), "test_deals": len(test), "features": NUMERIC + LEAGUES,
               "excluded_deals": 74 - len(data), "new_outgoing_xt_rows": len(extracted)}
    for name, (regressor, grid) in pipelines.items():
        encoding = ColumnTransformer([
            ("numeric", "passthrough", NUMERIC),
            ("league", OneHotEncoder(handle_unknown="ignore", sparse_output=False), LEAGUES),
        ])
        estimator = TransformedTargetRegressor(
            regressor=Pipeline([("features", encoding), ("model", regressor)]),
            func=np.log, inverse_func=np.exp,
        )
        search = GridSearchCV(estimator, grid, cv=LeaveOneOut(), scoring="neg_mean_absolute_error")
        search.fit(train[NUMERIC + LEAGUES], train.transfer_fee_eur / 1e6)
        test[name + "_prediction_eur"] = 1e6 * search.predict(test[NUMERIC + LEAGUES])
        summary[name] = {"params": search.best_params_, "training_cv_mae_eur_m": -search.best_score_}
        if name == "xgboost":
            fitted = search.best_estimator_.regressor_
            encoded = fitted["features"].transform(test[NUMERIC + LEAGUES])
            fitted["model"].save_model(OUTPUT / "model.json")
            restored = XGBRegressor()
            restored.load_model(OUTPUT / "model.json")
            np.testing.assert_allclose(np.exp(restored.predict(encoded)) * 1e6, test.xgboost_prediction_eur)
            summary["xgboost"]["encoded_features"] = fitted["features"].get_feature_names_out().tolist()
            summary["xgboost"]["league_categories"] = dict(zip(LEAGUES, [
                values.tolist() for values in fitted["features"].named_transformers_["league"].categories_
            ]))
    for label, rows in {"all": test, "inbound": test[test[BUYER].eq(1)],
                        "outbound": test[test[BUYER].eq(0)]}.items():
        actual = rows.transfer_fee_eur / 1e6
        training_rows = train if label == "all" else train[train[BUYER].eq(int(label == "inbound"))]
        median = training_rows.transfer_fee_eur.median() / 1e6
        summary[label] = {"deals": len(rows), "median_baseline": metrics(actual, np.full(len(rows), median))}
        for name in pipelines:
            summary[label][name] = metrics(actual, rows[name + "_prediction_eur"] / 1e6)
    data.to_csv(OUTPUT / "model_deals.csv", index=False)
    test.to_csv(OUTPUT / "holdout_predictions.csv", index=False)
    (OUTPUT / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
