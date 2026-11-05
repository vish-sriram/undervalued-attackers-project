"""Score WhoScored player actions and price the screened 2025/26 attackers."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from socceraction.spadl.config import actiontypes, results

from attacker_history_metrics import normalize, regulation_minutes
from build_attacker_history import CACHE, get_successful_move_actions, reference_model, spadl
from fit_fee_international import appearances
from fit_fee_ridge_outgoings import FOTMOB, league_stats, player_birth, read
from predict_transfer_fee import predict_fees

OUT = Path(__file__).resolve().parent / "outputs/attacker_valuations_2026"
AS_OF = "2026-11-05"
LEAGUES = {"ENG-Premier League", "FRA-Ligue 1", "GER-Bundesliga"}
LEAGUE_NAMES = {47: "Premier League", 53: "Ligue 1", 54: "Bundesliga", 87: "LaLiga"}
COORDS = ["start_x", "start_y", "end_x", "end_y"]
SELLER_CHECKS = {
    1197030: ("Liverpool", "Premier League", "https://www.liverpoolfc.com/teams/mens-team/hugo-ekitike"),
    1408580: ("Strasbourg", "Ligue 1", "https://www.transfermarkt.com/joaquin-panichelli/profil/spieler/1064871"),
    1047670: ("Union Berlin", "Bundesliga", "https://www.fc-union-berlin.de/en/news/andrej-ili-remains-with-the-iron-ones-XMbVDG"),
    1215652: ("Nottingham Forest", "Premier League", "https://www.acmilan.com/en/news/articles/media/2026-08-31/official-statement-omari-hutchinson"),
    1106563: ("Newcastle United", "Premier League", "https://www.newcastleunited.com/en/news/nick-woltemade-completes-juventus-loan-switch"),
    942358: ("Lille", "Ligue 1", "https://sevillafc.es/actualidad/noticias/fichaje-felix-correia-2026"),
}
CONTRACT_CHECKS = {
    1215652: "https://www.transfermarkt.com/omari-hutchinson/profil/spieler/741238/1000/",
    1106563: "https://www.transfermarkt.com/nick-woltemade/transfers/spieler/455661/transfer_id/5903935",
    942358: "https://www.transfermarkt.com/losc-lille/berateruebersicht/verein/1082",
    1408580: "https://www.transfermarkt.com/joaquin-panichelli/profil/spieler/1064871",
    961995: "https://www.transfermarkt.com/bukayo-saka/profil/spieler/433177/1000/",
    737066: "https://www.mancity.com/news/mens/erling-haaland-10-year-manchester-city-contract-63872694",
    1021382: "https://www.chelseafc.com/en/news/article/joao-pedro-signs-new-chelsea-contract",
    843040: "https://www.chelseafc.com/en/news/article/pedro-neto-signs-new-chelsea-contract",
}


def player_sources(pool):
    stats = [r for r in league_stats() if r[0] == 2025]
    ids = {}
    for player in pool.itertuples():
        matches = {r[3] for r in stats if r[1] == player.league and normalize(r[2]) == normalize(player.name)}
        if len(matches) == 1:
            ids[player.pid] = matches.pop()
    with ThreadPoolExecutor(max_workers=5) as executor:
        births = {pid: executor.submit(player_birth, fid) for pid, fid in ids.items()}
        sources = {}
        for pid, fid in ids.items():
            try:
                birth = births[pid].result()
                sources[pid] = (fid, birth, read(FOTMOB / "identity_profiles" / f"{fid}.json")["data"],
                                [r for r in stats if r[3] == fid])
            except Exception as error:
                print(f"Unavailable profile {fid}: {error}", flush=True)
    return sources


def score_events(pool):
    names = set(pool.pid)
    model, cutoff, _ = reference_model()
    assert cutoff < "2025-08-01"
    game_rows, outputs, raw_counts = [], [], Counter()
    with gzip.open(OUT / "player_source_events.jsonl.gz", "wt", encoding="utf-8") as raw:
        for league in sorted(LEAGUES):
            for path in sorted((CACHE / f"{league}_2526").glob("*.json")):
                match = json.loads(path.read_text())
                if not match or not match.get("events"):
                    continue
                participants = {
                    int(player["playerId"]): (match[side]["name"], player["position"])
                    for side in ("home", "away") for player in match[side]["players"]
                    if int(player["playerId"]) in names
                }
                if not participants:
                    continue
                for event in match["events"]:
                    pid = event.get("playerId")
                    if pid is not None and int(pid) in participants:
                        raw_counts[int(pid)] += 1
                        raw.write(json.dumps({"league": league, "game_id": int(path.stem),
                                              "match_date": match["startDate"][:10], "event": event},
                                             ensure_ascii=False) + "\n")
                minutes = regulation_minutes(match)
                actions = spadl(path, league, 2025, match)
                eligible = get_successful_move_actions(actions)
                scored = eligible.dropna(subset=COORDS)
                own = actions[actions.player_id.isin(participants)].copy()
                own["xt_added"] = np.nan
                own.loc[own.index.intersection(scored.index), "xt_added"] = pd.Series(model.rate(scored), index=scored.index)
                own["xt_eligible"] = own.index.isin(eligible.index)
                own["league"] = league
                own["match_date"] = match["startDate"][:10]
                own["action_type"] = own.type_id.map(dict(enumerate(actiontypes)))
                own["result"] = own.result_id.map(dict(enumerate(results)))
                outputs.append(own)
                for pid, (team, position) in participants.items():
                    if minutes[pid] > 0:
                        game_rows.append({"player_id": pid, "league": league, "game_id": int(path.stem),
                                          "team": team, "minutes": minutes[pid], "position": position})
    actions = pd.concat(outputs, ignore_index=True)
    actions.insert(0, "player", actions.player_id.map(pool.set_index("pid").name))
    assert not actions.duplicated(["league", "game_id", "action_id"]).any()
    actions.to_csv(OUT / "player_actions_xt.csv.gz", index=False)
    games = pd.DataFrame(game_rows)
    games.to_csv(OUT / "player_match_minutes.csv", index=False)
    return games, actions, raw_counts


def build_inputs(pool, sources, games, actions, raw_counts):
    rows = []
    for player in pool.itertuples():
        g = games[games.player_id.eq(player.pid)]
        a = actions[actions.player_id.eq(player.pid)]
        eligible = a[a.xt_eligible]
        row = {"name": player.name, "whoscored_player_id": player.pid, "screening_league": player.league,
               "prior_season": "2025/26", "valuation_date": AS_OF, "whoscored_role": player.position,
               "is_winger": int(player.position not in {"FW", "FWC"}), "buying_club_league": "Premier League",
               "whoscored_minutes": g.minutes.sum(), "whoscored_matches": len(g),
               "raw_events": raw_counts[player.pid], "spadl_actions": len(a),
               "scored_xt_actions": eligible.xt_added.count(),
               "unscored_eligible_actions": eligible.xt_added.isna().sum(),
               "total_xt_added": eligible.xt_added.sum(min_count=1)}
        issues = []
        if row["unscored_eligible_actions"] or not row["scored_xt_actions"]:
            issues.append("Eligible xT actions missing scores")
        if player.pid not in sources:
            row["validation_status"] = "FotMob identity unavailable"
            rows.append(row)
            continue
        fid, birth, profile, stats = sources[player.pid]
        as_of = date.fromisoformat(AS_OF)
        age = as_of.year - birth.year - ((as_of.month, as_of.day) < (birth.month, birth.day))
        team = profile.get("primaryTeam") or {}
        league = profile.get("mainLeague") or {}
        expiry = (profile.get("contractEnd") or {}).get("utcTime", "")[:10]
        row.update(fotmob_player_id=fid, fotmob_profile_url=f"https://www.fotmob.com/players/{fid}",
                   age_at_transfer=age, birth_date=birth.isoformat(), selling_club=team.get("teamName"),
                   selling_club_league=LEAGUE_NAMES.get(league.get("leagueId"), league.get("leagueName")),
                   selling_contract_expiry=expiry, prior_season_minutes=sum(s[4] for s in stats))
        row["contract_source_url"] = CONTRACT_CHECKS.get(fid, row["fotmob_profile_url"]) if expiry else None
        if fid in SELLER_CHECKS:
            row["selling_club"], row["selling_club_league"], row["seller_source_url"] = SELLER_CHECKS[fid]
        if not (23 <= age <= 26) or abs(age - player.age) > 1:
            issues.append("Verified age differs from screen")
        if fid == 1106563:
            proof = read(OUT / "woltemade_stuttgart_xg_verification.json")
            assert proof["shot_map_events"] > 0 and proof["woltemade_shots"] == []
            row["xg_zero_stint_source_url"] = proof["source_url"]
            stats = [(*s[:5], 0.0) if s[1] == "GER-Bundesliga" and s[4] == 90 and s[5] is None else s for s in stats]
        if any(s[5] is None for s in stats):
            issues.append("FotMob xG absent for a league stint")
        else:
            row["total_xg"] = sum(s[5] for s in stats)
        for competition in set(g.league) | {s[1] for s in stats}:
            expected = sum(s[4] for s in stats if s[1] == competition)
            observed = g.loc[g.league.eq(competition), "minutes"].sum()
            if not expected or not observed or abs(observed - expected) > max(20, .02 * expected):
                issues.append(f"Minutes gap in {competition} ({observed:.1f} vs {expected:.0f})")
        if fid not in SELLER_CHECKS and (
            not team or team.get("onLoan") or not league.get("leagueId") or league.get("season") != "2026/2027"
        ):
            issues.append("Current selling club or league unverified")
        if not expiry or expiry <= AS_OF:
            issues.append("Current contract expiry unavailable or expired")
        else:
            row["years_until_selling_contract_expiry"] = (date.fromisoformat(expiry) - as_of).days / 365.25
        try:
            intl = appearances(fid, AS_OF)
            row.update(intl, tournament_appearances=sum(intl.values()))
        except (KeyError, ValueError) as error:
            issues.append(f"International history unresolved ({error})")
        row["validation_status"] = "; ".join(issues) if issues else "validated"
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-events", action="store_true", help="Reuse the validated event exports.")
    args = parser.parse_args()
    pool = pd.read_csv(OUT / "screened_players.csv")
    assert len(pool) == pool.pid.nunique() == 34
    with ThreadPoolExecutor(max_workers=1) as executor:
        sources = executor.submit(player_sources, pool)
        if args.reuse_events:
            games = pd.read_csv(OUT / "player_match_minutes.csv")
            actions = pd.read_csv(OUT / "player_actions_xt.csv.gz")
            with gzip.open(OUT / "player_source_events.jsonl.gz", "rt", encoding="utf-8") as raw:
                raw_counts = Counter(int(json.loads(line)["event"]["playerId"]) for line in raw)
        else:
            games, actions, raw_counts = score_events(pool)
        rows = build_inputs(pool, sources.result(), games, actions, raw_counts)
    verified = predict_fees(rows[rows.validation_status.eq("validated")])
    rows = rows.merge(verified[["whoscored_player_id", "predicted_fee_eur"]], how="left",
                      on="whoscored_player_id", validate="one_to_one")
    training = pd.read_csv(OUT.parent / "fee_model_international" / "deals_and_audit.csv")
    training = training[training.window.eq("summer_2025") & training.exclusion_reason.isna()
                        & training.xt_per90.notna() & training.buying_league_premier_league.eq(1)].copy()
    training["total_xg"] = training.xg_per90 * training.prior_season_minutes / 90
    minutes = pd.read_csv(OUT.parent / "fee_model_xgboost" / "model_deals.csv")
    training = training.merge(minutes[["transfer_id", "total_xt_added"]], validate="one_to_one")
    numeric = ["age_at_transfer", "years_until_selling_contract_expiry", "prior_season_minutes",
               "total_xg", "total_xt_added", "tournament_appearances"]
    bounds = training[numeric].agg(["min", "max"])
    rows["outside_training_range"] = rows.apply(
        lambda row: "; ".join(col for col in numeric if pd.notna(row[col])
                              and not bounds.at["min", col] <= row[col] <= bounds.at["max", col]), axis=1)
    rows = rows.sort_values("predicted_fee_eur", ascending=False, na_position="last")
    rows.to_csv(OUT / "player_valuations.csv", index=False)
    print(rows[["name", "selling_club", "selling_contract_expiry", "total_xt_added", "total_xg",
                "predicted_fee_eur", "validation_status"]].to_string(index=False))
    print(f"{len(verified)} valuations, {len(actions)} SPADL actions, {sum(raw_counts.values())} raw events")


if __name__ == "__main__":
    main()
