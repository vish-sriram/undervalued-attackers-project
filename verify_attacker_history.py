"""Check time cutoffs, identity cardinality, aggregates and fee eligibility."""
import json
from pathlib import Path
import duckdb

ROOT=Path(__file__).resolve().parent
c=duckdb.connect(str(ROOT/'attacker_transfer_history.duckdb'),read_only=True)
assert c.sql('select count(*),count(distinct player_id) from attacker_transfer_metadata').fetchall()[0]==(71,71)
assert c.sql('select count(*) from attacker_player_match_stats e join attacker_transfer_metadata t using(player_id) where e.date>=t.transfer_date').fetchall()[0][0]==0
assert c.sql('select count(*) from attacker_player_match_stats where minutes<=0 or minutes>90 or receptions<0 or shots<0').fetchall()[0][0]==0
assert c.sql('select count(*) from (select player_id,league,game_id,count(*) n from attacker_player_match_stats group by all having n>1)').fetchall()[0][0]==0
assert c.sql('select count(*) from attacker_transfer_features where injury_missed_league_matches_pct<0 or injury_missed_league_matches_pct>100').fetchall()[0][0]==0
assert c.sql("select count(*) from attacker_transfer_features where model_ready and (not positive_permanent_fee_eligible or seasons_with_event_data<>1 or not event_source_collection_complete or not minutes_reconciled_to_season_source)").fetchall()[0][0]==0
assert c.sql('select count(*) from attacker_player_match_stats e join attacker_transfer_metadata t using(player_id) where season_start<>cast(substr(t.window,-4) as int)-1').fetchall()[0][0]==0
assert c.sql('select count(*) from attacker_transfer_features where requested_prior_seasons<>1').fetchall()[0][0]==0
assert c.sql('select count(*) from attacker_transfer_features f join (select player_id,sum(minutes) AS minutes_sum,sum(receptions) AS receptions_sum from attacker_player_match_stats group by player_id) e using(player_id) where abs(f.total_minutes-e.minutes_sum)>0.000001 or f.total_receptions<>e.receptions_sum').fetchall()[0][0]==0
summary=json.loads((ROOT/'outputs/attacker_history/summary.json').read_text())
print(json.dumps({'verification':'passed','summary':summary},indent=2))
