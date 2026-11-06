"""Predict a Premier League transfer fee using the retained arrivals-only model.

Input CSV columns: age_at_transfer, years_until_selling_contract_expiry,
prior_season_minutes, total_xg, total_xt_added, is_winger (0 or 1),
tournament_appearances, selling_club_league. Extra columns are preserved.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

MODEL = (Path(__file__).resolve().parent / "outputs" / "fee_model_international"
         / "arrivals_only" / "season_totals_ridge.joblib")
NUMERIC_INPUTS = [
    "age_at_transfer", "years_until_selling_contract_expiry", "prior_season_minutes",
    "total_xg", "total_xt_added", "is_winger", "tournament_appearances",
]


def predict_fees(players):
    required = NUMERIC_INPUTS + ["selling_club_league"]
    missing = sorted(set(required) - set(players.columns))
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    if players.empty:
        raise ValueError("Provide at least one player.")

    inputs = players[required].copy()
    inputs[NUMERIC_INPUTS] = inputs[NUMERIC_INPUTS].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(inputs[NUMERIC_INPUTS]).all().all():
        raise ValueError("All numeric inputs must be finite.")
    nonnegative = ["years_until_selling_contract_expiry", "total_xg", "tournament_appearances"]
    if (inputs[nonnegative] < 0).any().any():
        raise ValueError("Contract years, xG and tournament appearances must be nonnegative.")
    if not inputs.is_winger.isin([0, 1]).all():
        raise ValueError("is_winger must be 0 or 1.")
    if not inputs.tournament_appearances.mod(1).eq(0).all():
        raise ValueError("Tournament appearances must be whole numbers.")
    if (inputs[["prior_season_minutes", "age_at_transfer"]] <= 0).any().any():
        raise ValueError("Prior-season minutes and age must be positive.")
    inputs["selling_club_league"] = inputs.selling_club_league.astype("string").str.strip()
    if inputs.selling_club_league.isna().any() or inputs.selling_club_league.eq("").any():
        raise ValueError("Provide a selling-club league for every player.")
    if "buying_club_league" in players and not players.buying_club_league.eq("Premier League").all():
        raise ValueError("This model requires a Premier League buyer.")

    inputs["ln_prior_season_minutes"] = np.log(inputs.prior_season_minutes)
    inputs["buying_club_league"] = "Premier League"
    model = load(MODEL)
    fees_millions = model.predict(inputs[list(model.feature_names_in_)])
    return players.assign(predicted_fee_eur=(fees_millions * 1e6).round(2))
