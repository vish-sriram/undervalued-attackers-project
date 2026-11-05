"""Evaluate pre-deal senior tournament experience in the summer fee models."""
import json
from datetime import date

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from compare_log_fee_models import estimator
from attacker_history_metrics import team_key
from fit_fee_hedonic import metrics
from fit_fee_ridge_outgoings import BASE, BUYER, FOTMOB, OUT, POSITIONS, ROOT, SELLER, league_stats, player_birth, read
from fit_fee_xgboost import LEAGUES, OUTPUT as XT_OUTPUT

OUTPUT = ROOT / "outputs/fee_model_international"
EXCLUSIONS = {
    "Tommy Watson": ("Fee agreed before end of feature season", "https://www.brightonandhovealbion.com/media-article/mft-transfer-news-tommy-watson-transfer-complete-may-2025"),
    "Jørgen Strand Larsen": ("Prior loan clause", "https://www.wolves.co.uk/news/mens-first-team/20240702-strand-larsen-completes-wolves-loan-move/"),
    "Neal Maupay": ("Prior loan obligation", "https://www.bbc.co.uk/sport/football/articles/c93pwwwy2dyo"),
    "Emmanuel Emegha": ("Fee agreed before feature season ended", "https://www.chelseafc.com/en/news/article/chelsea-agree-deal-to-sign-emanuel-emegha"),
    "Donyell Malen": ("Prior loan option; permanent deal confirmed in May", "https://www.asroma.com/en/news/75420/donyell-malen-statement"),
    "Rasmus Højlund": ("Permanent conversion of existing loan", "https://www.manutd.com/en/news/uniteds-summer-transfers-ins-and-outs"),
    "Dwight McNeil": ("Linked player exchange", "https://www.evertonfc.com/teams/men/brennanjohnson"),
    "Brennan Johnson": ("Linked player exchange", "https://www.evertonfc.com/teams/men/brennanjohnson"),
}
AGREEMENTS = {"Anthony Gordon": ("2026-05-29", "https://www.fcbarcelona.com/en/football/first-team/news/4511134/anthony-gordon-becomes-an-fc-barcelona-player/featured")}
TOURNAMENTS = {77: "world_cup", 44: "copa_america", 50: "euro"}
TOURNAMENT_COLUMNS = [name + "_appearances" for name in TOURNAMENTS.values()]
WORLD_CUP_TEAM_ENDS = {8263: "2026-07-10", 6709: "2026-06-30"}


def add_arrivals(data, identities):
    stats = list(league_stats())
    records = []
    for deal in read(ROOT / "outputs/transfermarkt/premier_league_transfers.json"):
        tid, pid = int(deal["transfer_id"]), int(deal["player_id"])
        if tid in set(data.transfer_id) or pid in set(data.player_id):
            continue
        if (deal["position"] not in POSITIONS or deal["transfer_type"] != "Permanent / unspecified"
                or not deal["fee_eur"] or deal["fee_eur"] <= 0 or not deal["old_contract_expiry"]
                or not deal["old_club_league"] or pid not in identities):
            continue
        year = int(deal["window"][-4:]) - 1
        matches = [r for r in stats if r[0] == year and r[3] == identities[pid]]
        if not matches or any(r[5] is None for r in matches):
            continue
        minutes = sum(r[4] for r in matches)
        date_left = (date.fromisoformat(deal["old_contract_expiry"]) - date.fromisoformat(deal["transfer_date"])).days
        if minutes <= 0 or date_left < 0:
            continue
        records.append(dict(transfer_id=tid, player_id=pid, name=deal["player"], window=deal["window"],
                            season_start=year, transfer_date=deal["transfer_date"], position=deal["position"],
                            selling_club=deal["old_club"], buying_club=deal["new_club"],
                            selling_club_league=deal["old_club_league"], buying_club_league="Premier League",
                            transfer_fee_eur=deal["fee_eur"], prior_season_minutes=minutes,
                            age_at_transfer=deal["age_at_transfer"], selling_contract_expiry=deal["old_contract_expiry"],
                            years_until_selling_contract_expiry=date_left / 365.25,
                            xg_per90=90 * sum(r[5] for r in matches) / minutes,
                            ln_prior_season_minutes=np.log(minutes), transfer_source_url=deal["source_url"],
                            restored_from_fotmob=True, **{BUYER: 1, SELLER: int(deal["old_club_league"] == "Premier League")}))
    return pd.DataFrame(records)


def appearances(fid, cutoff):
    profile = read(FOTMOB / "identity_profiles" / f"{fid}.json")["data"]
    career = profile["careerHistory"]["careerItems"].get("national team", {})
    counts = dict.fromkeys(TOURNAMENT_COLUMNS, 0)
    seen = set()
    for season in (career.get("seasonEntries") or []):
        for tournament in season["tournamentStats"]:
            league = tournament.get("leagueId")
            if league not in TOURNAMENTS:
                continue
            year = int(tournament["seasonName"])
            key = season["teamId"], league, year
            if key in seen:
                raise ValueError(f"Duplicate tournament row {fid}: {key}")
            seen.add(key)
            if year >= int(cutoff[:4]):
                if league != 77 or year != 2026 or cutoff <= "2026-06-11":
                    continue
                team_end = WORLD_CUP_TEAM_ENDS.get(season["teamId"], "2026-07-19")
                if cutoff <= team_end:
                    raise ValueError(f"Match-level World Cup cutoff needed for {fid} on {cutoff}")
            counts[TOURNAMENTS[league] + "_appearances"] += int(tournament["appearances"])
    return counts


def evaluate(data, label):
    train = data[data.window.eq("summer_2025")]
    test = data[data.window.eq("summer_2026")].copy()
    assert set(train.player_id).isdisjoint(test.player_id)
    numeric = BASE + ["is_winger"] + (["xt_per90"] if label == "xt_sample" else [])
    candidates = {
        "hedonic_ols": (LinearRegression(), {}),
        "ridge": (make_pipeline(StandardScaler(), Ridge()), {"regressor__model__ridge__alpha": [0.1, 1, 10, 100]}),
        "xgboost": (XGBRegressor(tree_method="hist", learning_rate=0.05, min_child_weight=3,
                                 reg_lambda=5, random_state=2026, n_jobs=1),
                    {"regressor__model__max_depth": [1, 2], "regressor__model__n_estimators": [25, 60, 120]}),
    }
    y = train.transfer_fee_eur / 1e6
    results = {"train_n": len(train), "holdout_n": len(test), "models": {}}
    for mode, numbers in [("no_international", numeric), ("international", numeric + ["tournament_appearances"])]:
        features = numbers + LEAGUES
        for name, (reg, grid) in candidates.items():
            model = estimator(reg, numbers)
            if grid:
                search = GridSearchCV(model, grid, cv=LeaveOneOut(), scoring="neg_mean_absolute_error")
                search.fit(train[features], y)
                model = search.best_estimator_
            else:
                model.fit(train[features], y)
            key = f"{name}_{mode}"
            test[key] = model.predict(test[features])
            if label == "xt_sample" and key == "ridge_international":
                dump(model, OUTPUT / f"{label}_ridge.joblib")
            results["models"][key] = {}
            for scope, rows in [("all", test), ("inbound", test[test[BUYER].eq(1)])]:
                results["models"][key][scope] = metrics(rows.transfer_fee_eur / 1e6, rows[key])
    test["median_baseline"] = test[BUYER].map(y.groupby(train[BUYER]).median())
    results["models"]["median_baseline"] = {
        scope: metrics(rows.transfer_fee_eur / 1e6, rows.median_baseline)
        for scope, rows in [("all", test), ("inbound", test[test[BUYER].eq(1)])]
    }
    test.to_csv(OUTPUT / f"{label}_predictions_eur_m.csv", index=False)
    return results


def main():
    OUTPUT.mkdir(exist_ok=True)
    verified = {int(r["player_id"]): int(r["fotmob_player_id"]) for r in
                read(ROOT / "data/player_history/verified_fotmob_identities.json") if r["verified"]}
    assert player_birth(664500).isoformat() == "1998-06-04"
    verified[325443] = 664500  # Birth date verified against the Transfermarkt profile.
    departures = pd.read_csv(OUT / "departures_audit.csv").query('status == "eligible"')
    verified.update(dict(zip(departures.player_id.astype(int), departures.fotmob_player_id.astype(int))))
    data = pd.read_csv(OUT / "model_deals.csv")
    data["restored_from_fotmob"] = False
    added = add_arrivals(data, verified)
    data = pd.concat([data, added], ignore_index=True)
    data["exclusion_reason"] = data.name.map({k: v[0] for k, v in EXCLUSIONS.items()})
    data["deal_audit_source"] = data.name.map({k: v[1] for k, v in (EXCLUSIONS | AGREEMENTS).items()})
    data["known_agreement_date"] = data.name.map({k: v[0] for k, v in AGREEMENTS.items()})
    data["feature_cutoff_date"] = data.known_agreement_date.fillna(data.transfer_date)
    data["fotmob_player_id"] = data.player_id.map(verified)
    data["international_source_url"] = data.fotmob_player_id.map(lambda fid: f"https://www.fotmob.com/players/{fid}")
    data["is_winger"] = data.position.isin(["Left Winger", "Right Winger"]).astype(int)
    for idx, row in data[data.exclusion_reason.isna()].iterrows():
        profile = read(FOTMOB / "identity_profiles" / f"{row.fotmob_player_id}.json")["data"]
        teams = profile["careerHistory"]["careerItems"]["senior"].get("teamEntries") or []
        previous_loan = any(
            (team.get("transferType") or {}).get("localizationKey") == "on_loan"
            and team_key(team["team"]) == team_key(row.buying_club)
            and str(team.get("startDate") or "9999")[:10] < row.feature_cutoff_date
            and str(team.get("endDate") or "")[:10] >= f"{row.season_start + 1}-01-01"
            for team in teams
        )
        if previous_loan:
            data.loc[idx, "exclusion_reason"] = "Permanent signing by the previous season's loan club"
            data.loc[idx, "deal_audit_source"] = row.international_source_url
            continue
        try:
            counts = appearances(int(row.fotmob_player_id), row.feature_cutoff_date)
        except ValueError as error:
            data.loc[idx, "exclusion_reason"] = str(error)
            continue
        for name, value in counts.items():
            data.loc[idx, name] = value
        if pd.notna(row.known_agreement_date):
            data.loc[idx, "years_until_selling_contract_expiry"] = (
                date.fromisoformat(row.selling_contract_expiry) - date.fromisoformat(row.feature_cutoff_date)
            ).days / 365.25
    data["tournament_appearances"] = data[TOURNAMENT_COLUMNS].sum(axis=1, min_count=3)
    xt = pd.read_csv(XT_OUTPUT / "model_deals.csv")[["transfer_id", "xt_per90"]]
    data = data.merge(xt, how="left", on="transfer_id", validate="one_to_one")
    data.to_csv(OUTPUT / "deals_and_audit.csv", index=False)
    usable = data[data.exclusion_reason.isna()]
    assert usable.player_id.is_unique and usable.tournament_appearances.notna().all()
    report = {"restored_arrivals": added.name.tolist(),
              "excluded": data.loc[data.exclusion_reason.notna(), ["name", "exclusion_reason"]].to_dict("records"),
              "expanded_sample": evaluate(usable, "expanded_sample"),
              "xt_sample": evaluate(usable[usable.xt_per90.notna()], "xt_sample")}
    (OUTPUT / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for label in ("expanded_sample", "xt_sample"):
        r = report[label]
        print(label, r["train_n"], r["holdout_n"])
        print(pd.DataFrame({name: [s["all"]["mae_eur_m"], s["all"]["r_squared_euro"], s["inbound"]["mae_eur_m"],
                                   s["inbound"]["r_squared_euro"]] for name,s in r["models"].items()},
                           index=["all_mae","all_r2","in_mae","in_r2"]).T.round(3))
    print("Restored", report["restored_arrivals"], "Excluded", report["excluded"])


if __name__ == "__main__":
    main()
