"""Expand the transfer-fee model to paid permanent Premier League departures."""
import concurrent.futures
import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dateutil.relativedelta import relativedelta
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
TM = ROOT / "data/transfermarkt"
FOTMOB = ROOT / "data/player_history/fotmob"
OUT = ROOT / "outputs/fee_model_with_outgoings"
POSITIONS = {"Centre-Forward", "Second Striker", "Left Winger", "Right Winger"}
FEE = re.compile(r"€([\d.,]+)([MK]?)")
DATE = r"\d{2}/\d{2}/\d{4}"
BASE = ["age_at_transfer", "years_until_selling_contract_expiry", "ln_prior_season_minutes", "xg_per90"]
SELLER = "selling_league_premier_league"
BUYER = "buying_league_premier_league"
TRANSFERMARKT_BIRTH_DATES = {"745191": "1999-08-31", "663244": "1999-08-01"}


def read(path):
    return json.loads(path.read_text())


def parse_date(text):
    return datetime.strptime(text, "%d/%m/%Y").date()


def normalize(name):
    normalized = unicodedata.normalize("NFKD", name.casefold().translate(str.maketrans("øł", "ol")))
    return "".join(c for c in normalized if c.isascii() and c.isalnum())


def player_birth(player_id):
    cache = FOTMOB / "identity_profiles" / f"{player_id}.json"
    if cache.exists():
        profile = read(cache)
    else:
        response = requests.get(f"https://www.fotmob.com/players/{player_id}", timeout=15)
        response.raise_for_status()
        script = BeautifulSoup(response.text, "html.parser").find("script", id="__NEXT_DATA__")
        profile = json.loads(script.string)["props"]["pageProps"]
        cache.write_text(json.dumps(profile, ensure_ascii=False))
    return datetime.fromisoformat(profile["data"]["birthDate"]["utcTime"][:10]).date()


def league_stats():
    for path in sorted(FOTMOB.glob("*_mins_played.json")):
        league, year = path.name.removesuffix("_mins_played.json").rsplit("_", 1)
        if year not in {"2024", "2025"}:
            continue
        xg_file = path.with_name(f"{league}_{year}_expected_goals.json")
        xg = {}
        if xg_file.exists():
            xg = {(r["ParticiantId"], r["TeamId"]): r["StatValue"]
                  for block in read(xg_file)["TopLists"] for r in block["StatList"]}
        seen = set()
        for block in read(path)["TopLists"]:
            for row in block["StatList"]:
                key = row["ParticiantId"], row["TeamId"]
                if key in seen:
                    continue
                seen.add(key)
                yield (int(year), league, row["ParticipantName"], key[0], row["StatValue"], xg.get(key))


def outgoing_deals():
    arrivals = read(ROOT / "outputs/transfermarkt/premier_league_transfers.json")
    seen = {str(row["transfer_id"]) for row in arrivals}
    details = read(TM / "departure_details.json")
    for year, clubs in read(TM / "raw_departures.json").items():
        for club in clubs:
            for row in club["rows"]:
                fee = FEE.fullmatch(row["fee"].strip())
                if row["position"] not in POSITIONS or not fee or not row.get("detail"):
                    continue
                deal = row["detail"].rsplit("/", 1)[-1]
                if deal in seen:
                    continue
                seen.add(deal)
                lines = details.get(row["detail"], {}).get("rows", [])
                competition = next((line.split("\t") for line in lines if "\tCompetition\t" in line), [])
                transfer = next((re.search(DATE, line) for line in lines if line.startswith("Transfer date")), None)
                expiry = next((re.search(DATE, line) for line in lines if line.startswith("Remaining contract")), None)
                age_line = next((line for line in lines if line.startswith("Age at time of transfer")), "")
                age = tuple(int(m.group(1)) if (m := re.search(rf"(\d+) {unit}", age_line)) else 0
                            for unit in ("year", "month", "day"))
                amount = float(fee[1].replace(",", ".")) * {"M": 1e6, "K": 1e3, "": 1}[fee[2]]
                date = parse_date(transfer[0]) if transfer else None
                contract = parse_date(expiry[0]) if expiry else None
                detail_fee = next((line.split()[-1] for line in lines if line.startswith("Fee\n")), None)
                if amount <= 0:
                    continue
                if not competition or not date or detail_fee != fee[0] or not age[0]:
                    status = "transfer detail unavailable or disagrees with listing"
                elif date.year != int(year) or date.month not in range(6, 10):
                    status = "outside summer transfer dates"
                elif competition[0] != "Premier League" or competition[2] == "Premier League":
                    status = "outside Premier League to non-Premier League scope"
                elif not contract or contract < date:
                    status = "old contract expiry unavailable or invalid"
                else:
                    status = "pending prior-season data"
                yield {
                    "transfer_id": deal, "player_id": row["player_url"].split("/")[-1],
                    "name": row["name"], "window": f"summer_{year}", "transfer_date": date,
                    "season_start": int(year) - 1, "age_at_transfer": age[0],
                    "reported_age": age, "selling_club": club["seller"], "buying_club": row["buyer"],
                    "selling_club_league": competition[0] if competition else None,
                    "buying_club_league": competition[2] if competition else None,
                    SELLER: 1, BUYER: 0, "position": row["position"], "transfer_fee_eur": amount,
                    "selling_contract_expiry": contract,
                    "years_until_selling_contract_expiry": (contract-date).days/365.25 if contract and date else None,
                    "transfer_source_url": "https://www.transfermarkt.com" + row["detail"],
                    "profile_source_url": "https://www.transfermarkt.com" + row["player_url"],
                    "status": status,
                }


def add_performance(deals):
    stats = list(league_stats())
    matches = {}
    for deal in deals:
        if deal["status"] != "pending prior-season data":
            continue
        names = {normalize(deal["name"])}
        matches[deal["transfer_id"]] = [
            row for row in stats if row[0] == deal["season_start"] and normalize(row[2]) in names
        ]
    ids = {row[3] for rows in matches.values() for row in rows}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        tasks = {player_id: pool.submit(player_birth, player_id) for player_id in ids}
        births = {}
        for player_id, task in tasks.items():
            try:
                births[player_id] = task.result()
            except (requests.RequestException, ValueError, KeyError, AttributeError, TypeError) as error:
                print(f"FotMob identity unavailable for {player_id}: {error}")
    for deal in deals:
        if deal["transfer_id"] not in matches:
            continue
        rows = matches[deal["transfer_id"]]
        identified = set()
        profile_birth = TRANSFERMARKT_BIRTH_DATES.get(deal["player_id"])
        for player_id in {row[3] for row in rows} & births.keys():
            age = relativedelta(deal["transfer_date"], births[player_id])
            if profile_birth and births[player_id].isoformat() == profile_birth:
                identified.add(player_id)
            elif not profile_birth and (age.years, age.months, age.days) == deal["reported_age"]:
                identified.add(player_id)
        if len(identified) != 1:
            deal["status"] = ("no name match in prior-season FotMob cache" if not rows
                              else "identity not uniquely verified by name and exact age")
            continue
        player_id = identified.pop()
        rows = [row for row in rows if row[3] == player_id]
        deal["fotmob_player_id"] = player_id
        deal["performance_leagues"] = "; ".join(sorted({r[1] for r in rows}))
        deal["fotmob_birth_date"] = births[player_id]
        deal["identity_check"] = "profile birth date" if profile_birth else "exact age on transfer date"
        deal["prior_season_minutes"] = sum(r[4] for r in rows)
        if not deal["prior_season_minutes"] or any(r[5] is None for r in rows):
            deal["status"] = "prior-season xG or minutes unavailable"
            continue
        deal["xg_per90"] = 90 * sum(r[5] for r in rows) / deal["prior_season_minutes"]
        deal["status"] = "eligible"


def score(actual, predicted):
    return {
        "n": len(actual),
        "mae_eur_m": mean_absolute_error(actual, predicted),
        "rmse_eur_m": np.sqrt(np.mean((actual-predicted)**2)),
        "pearson_r": np.corrcoef(actual, predicted)[0, 1],
        "r_squared_euro": r2_score(actual, predicted),
        "r_squared_log": r2_score(np.log(actual), np.log(predicted)),
    }


def model(alpha=1):
    return TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), Ridge(alpha=alpha)),
                                      func=np.log, inverse_func=np.exp)


def fit(data, training, testing, columns):
    x = data[columns]
    fee = data.transfer_fee_eur / 1e6
    grid = GridSearchCV(model(), {"regressor__ridge__alpha": [0.01, 0.1, 1, 10, 100]},
                        cv=LeaveOneGroupOut(), scoring="neg_mean_absolute_error")
    grid.fit(x.loc[training], fee.loc[training], groups=data.loc[training, "player_id"])
    alpha = grid.best_params_["regressor__ridge__alpha"]
    predictions = grid.predict(x.loc[testing])
    final = model(alpha).fit(x, fee)
    scaler, ridge = final.regressor_.steps[0][1], final.regressor_.steps[1][1]
    slopes = ridge.coef_ / scaler.scale_
    intercept = ridge.intercept_ - scaler.mean_ @ slopes
    np.testing.assert_allclose(np.exp(intercept + x @ slopes), final.predict(x), rtol=1e-12)
    return alpha, predictions, dict(zip(["intercept", *columns], [intercept, *slopes]))


def main():
    OUT.mkdir(exist_ok=True)
    outgoing = list(outgoing_deals())
    add_performance(outgoing)
    audit = pd.DataFrame(outgoing)
    audit.to_csv(OUT / "departures_audit.csv", index=False)
    new = audit[audit.status.eq("eligible")].copy()

    inbound = pd.read_csv(ROOT / "outputs/attacker_history/model_ready_features.csv", dtype={"player_id": str})
    arrivals = pd.DataFrame(read(ROOT / "outputs/transfermarkt/premier_league_transfers.json"))
    arrivals["transfer_id"] = arrivals.transfer_id.astype(str)
    inbound["transfer_id"] = inbound.transfer_source_url.str.rsplit("/", n=1).str[-1]
    inbound["buying_club"] = inbound.transfer_id.map(arrivals.set_index("transfer_id").new_club)
    inbound["buying_club_league"] = "Premier League"
    inbound["prior_season_minutes"] = inbound.xg_minutes_fotmob
    inbound[SELLER] = inbound.selling_club_league.eq("Premier League").astype(int)
    inbound[BUYER] = 1
    inbound["season_start"] = inbound.window.str[-4:].astype(int) - 1
    assert len(inbound) == 37
    columns = ["transfer_id", "player_id", "name", "window", "season_start", "transfer_date", "position",
               "selling_club", "selling_club_league", "selling_contract_expiry", "transfer_source_url",
               "buying_club", "buying_club_league",
               "transfer_fee_eur", "prior_season_minutes", *[c for c in BASE if c != "ln_prior_season_minutes"],
               SELLER, BUYER]
    data = pd.concat([inbound[columns], new[columns]], ignore_index=True)
    data["ln_prior_season_minutes"] = np.log(data.prior_season_minutes)
    assert data.transfer_id.is_unique and (data.transfer_fee_eur > 0).all() and data.buying_club.notna().all()
    assert np.isfinite(data[BASE + [SELLER, BUYER]]).all().all()
    assert (data.season_start == data.window.str[-4:].astype(int) - 1).all()
    data.to_csv(OUT / "model_deals.csv", index=False)
    incoming = data[BUYER].eq(1)
    testing = data.window.eq("summer_2026")
    old_window = data.window.eq("summer_2025")
    training = old_window & ~data.player_id.isin(data.loc[testing, "player_id"])
    assert (training & ~incoming).any() and (testing & ~incoming).any()
    assert len(set(data.loc[training, "player_id"]) & set(data.loc[testing, "player_id"])) == 0
    test_data = data.loc[testing].copy()
    actual = test_data.transfer_fee_eur / 1e6
    inbound_test = test_data[BUYER].eq(1)
    summary = {"deals": len(data), "unique_players": data.player_id.nunique(),
               "target": "natural log of the fee in millions of euros",
               "minutes_and_xg_source": "FotMob domestic league season totals",
               "training_window": "summer_2025", "holdout_window": "summer_2026",
               "outgoing_deals": len(new), "training_deals": int(training.sum()),
               "training_removed_for_player_overlap": int((old_window & ~training).sum()),
               "holdout_deals": int(testing.sum()),
               "outgoing_audit_counts": audit.groupby(["window", "status"]).size().unstack(fill_value=0).to_dict("index")}
    for label, terms in {"seller_dummy": BASE + [SELLER],
                         "seller_and_buyer_dummies": BASE + [SELLER, BUYER]}.items():
        alpha, pred, coefficients = fit(data, training, testing, terms)
        test_data[label + "_prediction_eur"] = pred * 1e6
        summary[label] = {"alpha": alpha, "final_coefficients": coefficients,
                          "holdout_all": score(actual, pred),
                          "holdout_inbound": score(actual[inbound_test], pred[inbound_test]),
                          "holdout_outbound": score(actual[~inbound_test], pred[~inbound_test])}
    _, baseline, _ = fit(data.loc[incoming], old_window & incoming, testing & incoming, BASE)
    summary["inbound_only_same_minutes_source"] = score(actual[inbound_test], baseline)
    test_data.to_csv(OUT / "holdout_predictions.csv", index=False)
    (OUT / "model_and_metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
