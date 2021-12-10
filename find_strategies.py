import os
import pandas as pd
from scipy.optimize import minimize, Bounds, shgo, brute, basinhopping, differential_evolution
from strategy import Strategy
import json
import numpy as np
import optuna
optuna.logging.set_verbosity(optuna.logging.ERROR)


if __name__ == '__main__':
    df = Strategy.load_dataset('db.csv')
    df = df.rename(columns={
        f"bet365_1X2 Full Time_outcome_{i}_closing_value": f"bet365_{i}" for i in range(1, 4)
    })
    df['result'] = df[['score_ft_1', 'score_ft_2']].apply(lambda x: Strategy.cpt_winner(x[0], x[1]), axis=1)
    s = Strategy(matches=df)
    rank_features = [x for x in df.columns if "rank" in x]
    home_rank_features = [x for x in rank_features if '_1' in x and not '_A_' in x]
    away_rank_features = [x for x in rank_features if '_2' in x and not '_H_' in x]
    
    for x in home_rank_features:
        for y in away_rank_features:
            for result in [1, 2, 3]:
                path = f'strategies/strategy_{x}__{y}__{result}.json'
                if not os.path.exists(path):
                    s.look_for_strategy(df, x, y, result, 2000)
                    print(f"{path} Done.")
                else:
                    print(f"already existing file {path}")
    #s.look_for_strategies(df, '3M_H_P_rank_coeff_1', '3M_A_P_rank_coeff_2', 2000)
    #s.look_for_strategies(df, '3M_P_rank_coeff_1', '3M_P_rank_coeff_2', 2000)
    #s.look_for_strategies(df, '3M_G_rank_coeff_1', '3M_GA_rank_coeff_2', 2000)
    #s.look_for_strategies(df, '3M_H_GA_rank_coeff_1', '3M_A_G_rank_coeff_2', 2000)

