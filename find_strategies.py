import os
import pandas as pd
import json
import random 
from tqdm import tqdm
from strategy import Strategy

N_ITER = 2500
VERBOSE = True
NO_ODDS = True
PRIORITY_LEAGUES_FILE = "files/prio_leagues.csv"
if __name__ == '__main__':
    df = Strategy.load_dataset('sisi3.csv')
    priority = pd.read_csv(PRIORITY_LEAGUES_FILE, sep=';')
    priority = priority[priority["Prioritaire"] == 'OUI'][['Country', 'League']].apply(tuple, 1)
    df['country/league'] = df[['country', 'league']].apply(tuple, 1)
    df = df[df['country/league'].isin(priority)]

    #df = df.rename(columns={
    #    f"bet365_1X2 Full Time_outcome_{i}_closing_value": f"bet365_{i}" for i in range(1, 4)
    #})
    df['result'] = df[['score_ft_1', 'score_ft_2']].apply(lambda x: Strategy.cpt_winner(x[0], x[1]), axis=1)
    df['result_UO'] = df[['score_ft_1', 'score_ft_2']].apply(lambda x: 'U' if (x[0] + x[1]) < 2.5 else 'O', axis=1)
    rank_features = [x for x in df.columns if "rank" in x]
    home_rank_features = [x for x in rank_features if '_1' in x and not '_A_' in x]
    away_rank_features = [x for x in rank_features if '_2' in x and not '_H_' in x]
    print(rank_features)
    couples = [(x, y, z) for x in home_rank_features for y in away_rank_features for z in ['1', '2', '3', 'O', 'U']]
    couples = list(filter(
        lambda x: not os.path.exists(Strategy.get_strategy_path(x[0], x[1], x[2], NO_ODDS)),
        couples
    ))
    random.shuffle(couples)
    print(len(couples), " STRATEGIES to compute")
    for i in tqdm(range(len(couples))):
            x, y, z = couples[i]
            print(f"[DEBUG] Looking for the optimal strategy with Feature 1 {x}:, Feature 2: {y}, result: {z}...")
            path = Strategy.get_strategy_path(x, y, z, NO_ODDS)
            print(path)
            if not os.path.exists(path):
                if NO_ODDS:
                    Strategy.look_for_strategy(df, x, y, z, N_ITER, VERBOSE)
                else:
                    Strategy.look_for_strategy_2(df, x, y, z, N_ITER, VERBOSE)
                print(f"{path} Done.")
            else:
                print(f"already existing file {path}")