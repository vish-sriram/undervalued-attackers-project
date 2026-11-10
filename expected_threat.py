import json
from pathlib import Path

import numpy as np
import pandas as pd
from socceraction.data.opta.loader import _eventtypesdf
from socceraction.data.opta.parsers import WhoScoredParser
from socceraction.spadl.opta import convert_to_actions
from socceraction.xthreat import get_successful_move_actions, load_model

SURFACE = (
    Path(__file__).resolve().parent / "data/player_history/xt_reference_2024_25.json"
)
COORDINATES = ["start_x", "start_y", "end_x", "end_y"]


def score_actions(actions):
    eligible = get_successful_move_actions(actions)
    usable = eligible.dropna(subset=COORDINATES)
    scores = pd.Series(np.nan, index=actions.index, name="xt_added")
    scores.loc[usable.index] = load_model(str(SURFACE)).rate(usable)
    return scores


def read_match_actions(path, league, season):
    path = Path(path)
    home_id = int(json.loads(path.read_text())["home"]["teamId"])
    parser = WhoScoredParser(
        str(path), competition_id=league, season_id=str(season), game_id=int(path.stem)
    )
    events = pd.DataFrame.from_dict(parser.extract_events(), orient="index")
    events = events.merge(_eventtypesdf, on="type_id", how="left")
    actions = convert_to_actions(events, home_team_id=home_id)

    away = actions.team_id.ne(home_id)
    actions.loc[away, ["start_x", "end_x"]] = (
        105 - actions.loc[away, ["start_x", "end_x"]]
    )
    actions.loc[away, ["start_y", "end_y"]] = (
        68 - actions.loc[away, ["start_y", "end_y"]]
    )
    return actions.assign(xt_added=score_actions(actions))
