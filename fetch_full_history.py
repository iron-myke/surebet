import pandas as pd
import numpy as np
import pickle
import os

from betsapi import get_league_seasons
#from strategy import Strategy
from league import League

if __name__ == '__main__':
    leagues = pd.read_csv('tracked_leagues_final.csv')
    leagues_not_arjel = pd.read_csv('tracked_leagues_not_arjel.csv')
    leagues = pd.concat([leagues, leagues_not_arjel])
    for i, league in leagues.iterrows():
        print(league)
        df = get_league_seasons(league.id)
        league_path = League.get_league_path(league.league, league.country, '20-21')
        print(league_path)
        if os.path.exists(league_path):
            continue
        for season, g in df.groupby('season'):
            print(g.date.min(), g.date.max())
            g["country"] = league.country
            g["league"] = league.league
            league_path = League.get_league_path(league.league, league.country, g.season.iloc[0])
            if os.path.exists(league_path):
                continue
            _league = League(g, path=league_path)
            pickle.dump({
                        "matches": _league._matches,
                        "rankings": _league._ranking_by_date
                    }, open(league_path.replace('.csv', '.pkl'), "wb+"))
