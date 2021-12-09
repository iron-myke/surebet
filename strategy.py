import os
import pandas as pd
from scipy.optimize import minimize, Bounds, shgo, brute, basinhopping, differential_evolution
import json
import numpy as np
import optuna

FOLDER = "../leagues"
class Strategy:

    def __init__(self, matches):
        matches = matches.sort_values(by='str_date').reset_index(drop=True)
        # for s in ['strategy', 'strategy_2', 'strategy_3']:
        #     strategy = self.load_strategy_from_file(f"strategies/{s}.json")
        #     revenue = self.compute_revenue(matches, strategy)
        #     selected_matches = self.filter_matches(matches, strategy)
        #     print(revenue, revenue / len(selected_matches), len(selected_matches))
        #     selected_matches.to_csv(f'{s}.csv')


    @staticmethod
    def filter_matches(matches, strategy):
        cond = pd.Series([True for k in range(len(matches))])
        for k, v in strategy.items():
            if k == 'result':
                continue
            #print(k, v)
            if len(v) == 2:
                cond = cond & (matches[k] >= v[0]) & (matches[k] <= v[1]) & matches[k].notnull()
            else:
                cond = cond & (matches[k] == v[0])
        selected_matches = matches[cond]
        return selected_matches

    def compute_revenue(self, matches, strategy):
        filtered_matches = Strategy.filter_matches(matches, strategy)
        if len(filtered_matches) == 0:
            return -1e4
        predicted_result = strategy['result']
        gain = (filtered_matches[filtered_matches.result == predicted_result][f"bet365_{predicted_result}"] - 1)
        loss = filtered_matches[filtered_matches.result != predicted_result]
        return gain.sum() - len(loss)

    @staticmethod
    def load_strategy_from_file(filename="strategy.json"):
        with open(filename, 'r') as f:
            strategy = json.load(f)
        return strategy
    
    def look_for_strategies(self, matches, field_1, field_2, n_trials):
        for result in [1, 2, 3]:
            def test(x):
                strategy = {   
                    field_1: [x[0], x[1]],
                    field_2: [x[2], x[3]],
                    f"bet365_{result}": [x[4], x[5]],
                    "result": result
                }
                return self.compute_revenue(matches, strategy)

            def objective(trial):
                x = np.zeros(6)
                x[0] = trial.suggest_int(f'{field_1}_L', 1, 25)
                x[1] = trial.suggest_int(f'{field_1}_H', 1, 28)
                x[2] = trial.suggest_int(f'{field_2}_L', 1, 25)
                x[3] = trial.suggest_int(f'{field_2}_H', 1, 28)
                x[4] = trial.suggest_float('odd_L', 1.1, 10, step=0.01)
                x[5] = trial.suggest_float('odd_H', 1.2, 10, step=0.01)
                return test(x)
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
            strategy = {
                field_1: [study.best_params[f"{field_1}_L"], study.best_params[f"{field_1}_H"]],
                field_2: [study.best_params[f"{field_2}_L"], study.best_params[f"{field_2}_H"]],
                "result": result,
                f"bet365_{result}": [study.best_params["odd_L"], study.best_params["odd_H"]]
            }
            json.dump(strategy, open(f'strategies/strategy_{field_1}__{field_2}__{result}.json', 'w'))
    
    @staticmethod        
    def cpt_winner(g1, g2):
        if g1 > g2:
            return 1
        elif g1 < g2:
            return 3
        else:
            return 2
            
    
    @staticmethod
    def load_dataset():
        files = os.listdir(FOLDER)
        df = None
        for f in files:
            df = pd.concat([df, pd.read_csv(f"{FOLDER}/{f}")])
        return df.reset_index(drop=True)

    @staticmethod
    def analyze_strategy(strategy, matches):
        filtered_matches = Strategy.filter_matches(matches, strategy)
        filtered_matches.loc[:, "year"] = pd.to_datetime(filtered_matches.loc[:, 'date']).dt.year
        predicted_result = strategy.get('result', -1)
        filtered_matches["gain"] = 0
        filtered_matches.loc[filtered_matches[filtered_matches.result == predicted_result].index, "gain"] = filtered_matches.loc[filtered_matches[filtered_matches.result == predicted_result].index, f"bet365_{predicted_result}"]
        filtered_matches.gain = filtered_matches.gain - 1
        filtered_matches["cum_gain"] = filtered_matches.gain.cumsum()
        gain_by_year = filtered_matches.groupby('year')["gain"].sum()
        return filtered_matches, gain_by_year


if __name__ == '__main__':
    df = Strategy.load_dataset()
    df = df.rename(columns={
        f"bet365_1X2 Full Time_outcome_{i}_closing_value": f"bet365_{i}" for i in range(1, 4)
    })
    df['result'] = df[['score_ft_1', 'score_ft_2']].apply(lambda x: Strategy.cpt_winner(x[0], x[1]), axis=1)
    s = Strategy(matches=df)
    s.look_for_strategies(df, '3M_H_P_rank_coeff_1', '3M_A_P_rank_coeff_2', 2000)
    s.look_for_strategies(df, '3M_P_rank_coeff_1', '3M_P_rank_coeff_2', 2000)
    s.look_for_strategies(df, '3M_G_rank_coeff_1', '3M_GA_rank_coeff_2', 2000)
    s.look_for_strategies(df, '3M_H_GA_rank_coeff_1', '3M_A_G_rank_coeff_2', 2000)


