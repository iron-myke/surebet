import json
import numpy as np
import optuna
import os
import pandas as pd
import traceback
import tqdm

#from scipy.optimize import minimize, Bounds, shgo, brute, basinhopping, differential_evolution

optuna.logging.set_verbosity(optuna.logging.ERROR)
CONFIG_FILEPATH = "config.json"
CONFIG = json.load(open(CONFIG_FILEPATH, 'r'))
LEAGUE_FOLDER = CONFIG.get("LEAGUE_FOLDER")
STRATEGY_FOLDER = CONFIG.get("STRATEGY_FOLDER")

class Strategy:

    @staticmethod
    def filter_matches(matches, strategy):
        matches["cond"]= pd.Series([True for k in range(len(matches))])
        for k, v in strategy.items():
            if k == 'result':
                continue
            if len(v) == 2:
                matches.cond = matches.cond & (matches[k].astype(float) >= v[0]) & (matches[k].astype(float) <= v[1]) & matches[k].notnull()
            else:
                matches.cond = matches.cond & (matches[k].astype(float) == v[0])
        return matches[matches.cond]

    
    @staticmethod
    def compute_revenue(matches, strategy):
        filtered_matches = Strategy.filter_matches(matches, strategy)
        if len(filtered_matches) == 0:
            return -1e4
        predicted_result = strategy['result']
        if predicted_result in ['1', '2', '3']:
            cond = filtered_matches.result == int(predicted_result)
        else:
            cond = filtered_matches.result_UO == predicted_result
        gain = filtered_matches[cond][f"bet365_{predicted_result}"] - 1
        loss = filtered_matches[~cond]
        revenue = gain.sum() - len(loss)
        print("REV:", revenue)
        print()
        return revenue

    @staticmethod
    def load_strategy_from_file(filename="strategy.json"):
        with open(filename, 'r') as f:
            strategy = json.load(f)
        return strategy

    @staticmethod
    def get_strategy_path(field_1, field_2, result, bis=True):
        if bis:
            return f"{STRATEGY_FOLDER}/strategy_{field_1}__{field_2}__{result}.json"
        return f"{STRATEGY_FOLDER}/strategy_no_odds_{field_1}__{field_2}__{result}.json"
    
    @staticmethod
    def save_strategy(strategy, filename=None):
        if filename:
            json.dump(strategy, open(filename, 'w'))
    
    @staticmethod
    def look_for_strategy(matches, field_1, field_2, result, n_trials=2000, verbose=False):
        def revenue(x):
            strategy = {   
                field_1: [x[0], x[1]],
                field_2: [x[2], x[3]],
                f"bet365_{result}": [x[4], x[5]],
                "result": result
            }
            rev = Strategy.compute_revenue(matches, strategy)
            return rev
        
        def objective(trial):
            x = np.zeros(6)
            x[0] = trial.suggest_int(f'{field_1}_L', 1, 25)
            x[1] = trial.suggest_int(f'{field_1}_H', 1, 28)
            x[2] = trial.suggest_int(f'{field_2}_L', 1, 25)
            x[3] = trial.suggest_int(f'{field_2}_H', 1, 28)
            x[4] = trial.suggest_float('odd_L', 1., 12, step=0.01)
            x[5] = trial.suggest_float('odd_H', 1.1, 12, step=0.01)
            return revenue(x)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)
        strategy = {
            field_1: [study.best_params[f"{field_1}_L"], study.best_params[f"{field_1}_H"]],
            field_2: [study.best_params[f"{field_2}_L"], study.best_params[f"{field_2}_H"]],
            "result": result,
            f"bet365_{result}": [study.best_params["odd_L"], study.best_params["odd_H"]]
        }
        Strategy.save_strategy(strategy, Strategy.get_strategy_path(field_1, field_2, result)) 

    @staticmethod
    def look_for_strategy_2(matches, field_1, field_2, result, n_trials=2000, verbose=False):
        def revenue(x):
            strategy = {   
                field_1: [x[0], x[1]],
                field_2: [x[2], x[3]],
                "result": result
            }
            return Strategy.compute_revenue(matches, strategy)

        def objective(trial):
            x = np.zeros(6)
            x[0] = trial.suggest_int(f'{field_1}_L', 1, 25)
            x[1] = trial.suggest_int(f'{field_1}_H', 1, 30)
            x[2] = trial.suggest_int(f'{field_2}_L', 1, 25)
            x[3] = trial.suggest_int(f'{field_2}_H', 1, 30)
            return revenue(x)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)
        strategy = {
            field_1: [study.best_params[f"{field_1}_L"], study.best_params[f"{field_1}_H"]],
            field_2: [study.best_params[f"{field_2}_L"], study.best_params[f"{field_2}_H"]],
            "result": result,
        }
        Strategy.save_strategy(strategy, Strategy.get_strategy_path(field_1, field_2, result, True)) 

    @staticmethod        
    def cpt_winner(g1, g2):
        if g1 > g2:
            return 1
        elif g1 < g2:
            return 3
        else:
            return 2

    
    @staticmethod
    def load_dataset(filename=None):
        if filename and os.path.exists(filename):
            print("Found file")
            return pd.read_csv(filename)
        files = [f for f in os.listdir(LEAGUE_FOLDER) if '.csv' in f]
        df = None
        for i in tqdm.tqdm(range(len(files))):
            f = files[i]
            try:
                _df = pd.read_csv(f"{LEAGUE_FOLDER}/{f}")
                _df = _df.replace({'-': np.nan})
                _x = _df[_df["bet365_1"]=='-']
                df = pd.concat([df, _df])
            except Exception:
                traceback.print_exc()
        df = df.reset_index(drop=True)
        for suffix in [1, 2, 3, 'U', 'O', 'ht_1', 'ht_2', 'ht_3']:
            df[f"bet365_{suffix}"] = df[f"bet365_{suffix}"].astype(float, errors='ignore')
            df[~df[f"bet365_{suffix}"].apply(lambda x: isinstance(x, float))] = np.nan
        if filename:
            df.to_csv(filename)
        return df

    @staticmethod
    def analyze_strategy(strategy, matches):
        filtered_matches = Strategy.filter_matches(matches, strategy)
        filtered_matches = filtered_matches.sort_values(by='date')
        filtered_matches.loc[:, "year"] = pd.to_datetime(filtered_matches.loc[:, 'date']).dt.year
        predicted_result = strategy.get('result', -1)
        filtered_matches["gain"] = 0
        filtered_matches.loc[filtered_matches[filtered_matches.result.astype(str) == predicted_result].index, "gain"] = filtered_matches.loc[filtered_matches[filtered_matches.result.astype(str) == predicted_result].index, f"bet365_{predicted_result}"]
        filtered_matches.gain = filtered_matches.gain - 1
        filtered_matches["cum_gain"] = filtered_matches.gain.cumsum()
        gain_by_year = filtered_matches.groupby('year')["gain"].sum()
        return filtered_matches, gain_by_year
