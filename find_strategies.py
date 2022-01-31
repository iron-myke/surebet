import os
import pandas as pd
import json
import random 
from strategy import Strategy

N_ITER = 7000
VERBOSE = True

if __name__ == '__main__':
    df = Strategy.load_dataset('db_prod_updated.csv')
    #df = df.rename(columns={
    #    f"bet365_1X2 Full Time_outcome_{i}_closing_value": f"bet365_{i}" for i in range(1, 4)
    #})
    df['result'] = df[['score_ft_1', 'score_ft_2']].apply(lambda x: Strategy.cpt_winner(x[0], x[1]), axis=1)
    rank_features = [x for x in df.columns if "rank" in x]
    home_rank_features = [x for x in rank_features if '_1' in x and not '_A_' in x]
    away_rank_features = [x for x in rank_features if '_2' in x and not '_H_' in x]
    couples = [(x,y) for x in home_rank_features for y in away_rank_features]
    random.shuffle(couples)
    print(len(home_rank_features) * len(away_rank_features) * 3, " STRATEGIES")
    for x, y in couples:
            print(x, y)
            for result in [1, 2, 3]:
                path = Strategy.get_strategy_path(x, y, result)
                if not os.path.exists(path):
                    Strategy.look_for_strategy(df, x, y, result, N_ITER, VERBOSE)
                    print(f"{path} Done.")
                else:
                    print(f"already existing file {path}")