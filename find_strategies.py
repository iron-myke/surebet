import os
import pandas as pd
import json

from strategy import Strategy

N_ITER = 4000

if __name__ == '__main__':
    df = Strategy.load_dataset('db.csv')
    df = df.rename(columns={
        f"bet365_1X2 Full Time_outcome_{i}_closing_value": f"bet365_{i}" for i in range(1, 4)
    })
    df['result'] = df[['score_ft_1', 'score_ft_2']].apply(lambda x: Strategy.cpt_winner(x[0], x[1]), axis=1)
    rank_features = [x for x in df.columns if "rank" in x]
    home_rank_features = [x for x in rank_features if '_1' in x and not '_A_' in x]
    away_rank_features = [x for x in rank_features if '_2' in x and not '_H_' in x]
    
    for x in home_rank_features:
        for y in away_rank_features:
            for result in [1, 2, 3]:
                path = Strategy.get_strategy_path(x, y, result)
                if not os.path.exists(path):
                    Strategy.look_for_strategy(df, x, y, result, N_ITER)
                    print(f"{path} Done.")
                else:
                    print(f"already existing file {path}")