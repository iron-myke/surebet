import json
import numpy as np
import os
import pandas as pd
import pickle
import time

CONFIG_FILEPATH = "config.json"
CONFIG = json.load(open(CONFIG_FILEPATH, 'r'))
LEAGUE_FOLDER = CONFIG.get("LEAGUE_FOLDER")
RAW_FILE = CONFIG.get("RAW_FILE")
PRIORITY_LEAGUES_FILE = CONFIG.get("PRIORITY_LEAGUES_FILE")
GA_COEFF_FILE = CONFIG.get("GA_COEFF_FILE")
POINTS_COEFF_FILE = CONFIG.get("POINTS_COEFF_FILE")


class League: 

    __FEATS = ["P", "G", "GA"]
    __SUFFIXES = ['', '_coeff']

    with open('files/home.json', 'r') as f:
        __HOME_MAPPING = json.load(f)
    with open('files/away.json', 'r') as f:
        __AWAY_MAPPING = json.load(f) 
    with open('files/features.json', 'r') as f:
        __FEATURES = json.load(f)["features"]

    def __init__(self, df, path, from_dict=None):
        start = time.time()
        self.name = f"{df.league.iloc[0]}_{df.country.iloc[0]}_{df.season.iloc[0]}".replace(' ', '_').replace('/', '_')
        self._load_coefficients()
        self.preprocessing(df)
        if from_dict:
            self._matches = from_dict.matches
            self._ranking_by_date = from_dict.rankings
        else:
            self.compute_ranking_by_date()
            self.compute_output()
        if path:
            self.output.to_csv(path)
        end = time.time()
        print(f"Done.")
        print(f"Time elapsed: {end - start}")
        print()
        #features = [x + y + z + a for x in['H_', 'A_', '',] for y in ['_3M', ''] for z in ['P', 'G', 'GA'] for a in ['', '_coeff']]


    def _load_coefficients(self):
        coeffs = pd.read_csv(POINTS_COEFF_FILE, sep=';')
        coeffs_gA =  pd.read_csv(GA_COEFF_FILE, sep=';')
        self._g_coeffs = {r.opponent_rank: r.coeff for i, r in coeffs.iterrows()}
        self._ga_coeffs = {r.opponent_rank: float(r.coeff.replace(',', '.')) for i, r in coeffs_gA.iterrows()}
        self._g_coeffs[0.0] = 1.0
        self._ga_coeffs[0.0] = 1.0

    def preprocessing(self, df):
        print('Preprocessing...')
        self._matches = df.drop_duplicates(['date', '1_team', '2_team'])
        self._teams = pd.concat([
            self._matches['1_team'].value_counts(), 
            self._matches['2_team'].value_counts()
        ], axis='columns')
        
        self._teams['n_matches'] = self._teams['1_team'] + self._teams['2_team']
        print(self._teams.n_matches)
        self._teams = self._teams.dropna()[self._teams.n_matches >= 9].index.values
        self.n_teams = len(self._teams)
        print("N_TEAMS: ", len(self._teams))
        self._matches = self._matches[
            self._matches['1_team'].isin(self._teams) & self._matches['2_team'].isin(self._teams)
        ]
        print('N_MATCHES :', len(self._matches))

        self._matches = self._matches.reset_index()
        self._matches = self._matches.sort_values(by='date', ascending='False')
        self._matches.date = pd.to_datetime(self._matches.date)
        self._matches['str_date'] = self._matches.date.dt.to_period('D')

        self._matches['1_Hn'] = self._matches.apply(lambda x: len(self._matches[(self._matches.date < x.date) & (self._matches['1_team'] == x['1_team'])]), axis=1)
        self._matches['1_An'] = self._matches.apply(lambda x: len(self._matches[(self._matches.date < x.date) & (self._matches['2_team'] == x['1_team'])]), axis=1)
        self._matches['1_n'] = self._matches['1_An'] + self._matches['1_Hn']

        self._matches['2_Hn'] = self._matches.apply(lambda x: len(self._matches[(self._matches.date < x.date) & (self._matches['1_team'] == x['2_team'])]), axis=1)
        self._matches['2_An'] = self._matches.apply(lambda x: len(self._matches[(self._matches.date < x.date) & (self._matches['2_team'] == x['2_team'])]), axis=1)
        self._matches['2_n'] = self._matches['2_An'] + self._matches['2_Hn']

        self._matches['3M_avg_rate_1'] = self._matches.apply(lambda x: (self._matches[(self._matches['1_team'] == x['1_team']) & (self._matches['1_n'] >= x['1_n'] - 3) & (self._matches['1_n'] < x['1_n'])]["Team1-note"].sum() +
            self._matches[(self._matches['2_team'] == x['1_team']) & (self._matches['1_n'] >= x['1_n'] - 3) & (self._matches['1_n'] < x['1_n'])]["Team2-note"].sum()) / 3, axis=1)
        self._matches['3M_avg_rate_2'] = self._matches.apply(lambda x: (self._matches[(self._matches['1_team'] == x['2_team']) & (self._matches['1_n'] >= x['1_n'] - 3) & (self._matches['1_n'] < x['1_n'])]["Team1-note"].sum() +
            self._matches[(self._matches['2_team'] == x['2_team']) & (self._matches['1_n'] >= x['1_n'] - 3) & (self._matches['1_n'] < x['1_n'])]["Team2-note"].sum()) / 3, axis=1)

        self._matches['1_pts'] = self._matches.apply(lambda x: League.cpt_points(x['score_ft_1'], x['score_ft_2']), axis=1)
        self._matches['2_pts'] = self._matches.apply(lambda x: League.cpt_points(x['score_ft_2'], x['score_ft_1']), axis=1)


    def compute_ranking_by_date(self):
        print('Computing Ranking by Date...')
        matches_by_date = self._matches.groupby('str_date')
        cum_points = pd.DataFrame.from_dict({t: {f:0 for f in League.__FEATURES} for t in self._teams}, orient="index")
        self._ranking_by_date = {}
        self._ranking_by_date[0] = cum_points
        last_date = 0  

        _l_g_coeffs = [('1_P_coeff', '3M_A_P_rank_2', '1_pts'), ('2_P_coeff', '3M_P_rank_1', '2_pts'),
            ('1_P_coeff_H', '3M_A_P_rank_2', '1_pts'), ('2_P_coeff_A', '3M_H_P_rank_1', '2_pts'),
            ('1_G_coeff', '3M_GA_rank_2', 'score_ft_1'), ('2_G_coeff', '3M_GA_rank_1', 'score_ft_2'),
            ('1_G_coeff_H', '3M_GA_rank_2', 'score_ft_1'), ('2_G_coeff_A', '3M_H_GA_rank_1', 'score_ft_2')]
        
        _l_ga_coeffs = [('1_GA_coeff', '3M_G_rank_2', 'score_ft_2'), ('2_GA_coeff', '3M_G_rank_1', 'score_ft_1'),
            ('1_GA_coeff_H', '3M_A_G_rank_2', 'score_ft_2'), ('2_GA_coeff_A', '3M_H_G_rank_1', 'score_ft_1')]
        l = [("P_rank", ["P", "DIFF", "G"], False), ("H_P_rank", ["H_P", "H_DIFF", "H_G"], False), 
                ("A_P_rank", ["A_P", "A_DIFF", "A_G"], False), ("G_rank", ["G"], False), 
                ("H_G_rank", ["H_G"], False), ("A_G_rank", ["A_G"], False), ("GA_rank", ["GA"], True), 
                ("H_GA_rank", ["H_GA"], True), ("A_GA_rank", ["A_GA"], True), 
                ("3M_P_rank", ["3M_P", "3M_DIFF", "3M_G"], False), ("3M_H_P_rank", ["H_3M_P", "H_3M_DIFF", "H_3M_G"], False), 
                ("3M_A_P_rank", ["A_3M_P", "A_3M_DIFF", "A_3M_G"], False), ("3M_G_rank", ["3M_G"], False), ("3M_H_G_rank", ["H_3M_G"], False),
                ("3M_A_G_rank", ["A_3M_G"], False), ("3M_A_G_rank", ["A_3M_G"], False), ("3M_GA_rank", ["3M_GA"], True), 
                ("3M_H_GA_rank", ["H_3M_GA"], True), ("3M_A_GA_rank", ["A_3M_GA"], True),
                ("3M_P_coeff_rank", ["3M_P_coeff", "3M_DIFF_coeff", "3M_G_coeff"], False), ("3M_H_P_coeff_rank", ["H_3M_P_coeff", "H_3M_DIFF_coeff", "H_3M_G_coeff"], False), 
                ("3M_A_P_coeff_rank", ["A_3M_P_coeff", "A_3M_DIFF_coeff", "A_3M_G_coeff"], False), ("3M_G_coeff_rank", ["3M_G_coeff"], False), ("3M_H_G_coeff_rank", ["H_3M_G_coeff"], False),
                ("3M_A_G_coeff_rank", ["A_3M_G_coeff"], False), ("3M_A_G_coeff_rank", ["A_3M_G_coeff"], False), ("3M_GA_coeff_rank", ["3M_GA_coeff"], True), 
                ("3M_H_GA_coeff_rank", ["H_3M_GA_coeff"], True), ("3M_A_GA_coeff_rank", ["A_3M_GA_coeff"], True),]

        for i, g in matches_by_date:
            cum_points = self._ranking_by_date[last_date].copy()
            if len(self._ranking_by_date.keys()) > 1:
                g = g.merge(cum_points, left_on='1_team', right_index=True)
                g = g.merge(cum_points, left_on='2_team', right_index=True, suffixes=('_1', '_2'))
                for i_3, j, k,  in _l_g_coeffs:
                    g[i_3] = g[[j, k]].apply(lambda x: (self._g_coeffs[x[0]] * x[1]) if not pd.isna(x[0]) else x[1], axis=1)
                for i_3, j, k in _l_ga_coeffs:
                    g[i_3] = g[[j, k]].apply(lambda x: (self._ga_coeffs[x[0]] * x[1]) if not pd.isna(x[0]) else x[1], axis=1)
            else:
                for i_3, j, k  in _l_g_coeffs:
                    g[i_3] = g[k]
                for i_3, j, k in _l_ga_coeffs:
                    g[i_3] = g[k]
    
            for i_m, m in g.iterrows():
                cum_points = self.match_features_update(m, cum_points)
            
            for suffix in League.__SUFFIXES:
                cum_points[f"DIFF{suffix}"] = cum_points[f"G{suffix}"] - cum_points[f"GA{suffix}"]
                cum_points[f"H_DIFF{suffix}"] = cum_points[f"H_G{suffix}"] - cum_points[f"H_GA{suffix}"]
                cum_points[f"A_DIFF{suffix}"] = cum_points[f"A_G{suffix}"] - cum_points[f"A_GA{suffix}"]
                cum_points[f"H_3M_DIFF{suffix}"] = cum_points[f"H_3M_G{suffix}"] - cum_points[f"H_3M_G{suffix}"]
                cum_points[f"3M_DIFF{suffix}"] = cum_points[f"3M_G{suffix}"] - cum_points[f"3M_GA{suffix}"]
                cum_points[f"A_3M_DIFF{suffix}"] = cum_points[f"A_3M_G{suffix}"] - cum_points[f"A_3M_GA{suffix}"]
            
            for r, args, order in l:
                cum_points[r] = cum_points[args].apply(tuple,axis=1).rank(method='min',ascending=order).astype(int)
            
            cum_points = cum_points.sort_values(by='P_rank')
            self._ranking_by_date[i] = cum_points
            last_date = i

    def match_features_update(self, m, cum_points):
        t_1, t_2 = m["1_team"], m["2_team"]
        for x, y in League.__HOME_MAPPING.items():
            cum_points.loc[t_1, x] += m[y] 
        for x, y in League.__AWAY_MAPPING.items():
            cum_points.loc[t_2, x] += m[y] 
        
        if m["1_n"] >= 3:
            _M3_match = self._matches[((self._matches['1_team'] == t_1) & (self._matches['1_n'] == m["1_n"] - 3)) | ((self._matches['2_team'] == t_1) & (self._matches['2_n'] == m["1_n"] - 3))]
            _3M_cum_points =self._ranking_by_date[_M3_match.str_date.iloc[0]]
            for suffix in League.__SUFFIXES:
                for feat in League.__FEATS:
                    cum_points.loc[t_1, f"3M_{feat}{suffix}"] = cum_points.loc[t_1, f"{feat}{suffix}"] - _3M_cum_points.loc[t_1, f"{feat}{suffix}"]
        else:
            for suffix in League.__SUFFIXES:
                for feat in League.__FEATS:
                    cum_points.loc[t_1, f"3M_{feat}{suffix}"] = cum_points.loc[t_1, f"{feat}{suffix}"]
                
        if m["1_Hn"] >= 3:
            _M3_home_match = self._matches[((self._matches['1_team'] == t_1) & (self._matches['1_Hn'] == m["1_Hn"] - 3))]
            _3M_cum_points =self._ranking_by_date[_M3_home_match.str_date.iloc[0]]
            for suffix in League.__SUFFIXES:
                for feat in League.__FEATS:
                    cum_points.loc[t_1, f"H_3M_{feat}{suffix}"] = cum_points.loc[t_1, f"H_{feat}{suffix}"] - _3M_cum_points.loc[t_1, f"H_{feat}{suffix}"]
        else:
            for suffix in League.__SUFFIXES:
                for feat in League.__FEATS:
                    cum_points.loc[t_1, f"H_3M_{feat}{suffix}"] = cum_points.loc[t_1, f"H_{feat}{suffix}"]

        if m["2_n"] >= 3:
            _M3_match = self._matches[((self._matches['1_team'] == t_2) & (self._matches["1_n"] == m["2_n"] - 3)) | ((self._matches['2_team'] == t_2) & (self._matches['2_n'] == m["2_n"] - 3))]
            _3M_cum_points =self._ranking_by_date[_M3_match.str_date.iloc[0]]
            for suffix in League.__SUFFIXES:
                for feat in League.__FEATS:
                    cum_points.loc[t_2, f"3M_{feat}{suffix}"] = cum_points.loc[t_2, f"{feat}{suffix}"] - _3M_cum_points.loc[t_2, f"{feat}{suffix}"]
        else:
            for suffix in League.__SUFFIXES:
                for feat in League.__FEATS:
                    cum_points.loc[t_2, f"3M_{feat}{suffix}"] = cum_points.loc[t_2, f"{feat}{suffix}"]
        
        if m["2_An"] >= 3:
            _M3_away_match = self._matches[((self._matches['2_team'] == t_2) & (self._matches["2_An"] == m["2_An"] - 3))]
            _3M_cum_points =self._ranking_by_date[_M3_away_match.str_date.iloc[0]]    
            for suffix in League.__SUFFIXES:
                for feat in League.__FEATS:
                    cum_points.loc[t_2, f"A_3M_{feat}{suffix}"] = cum_points.loc[t_2, f"A_{feat}{suffix}"] - _3M_cum_points.loc[t_2, f"A_{feat}{suffix}"]
        else:
            for suffix in League.__SUFFIXES: 
                for feat in League.__FEATS:
                    cum_points.loc[t_2, f"A_3M_{feat}{suffix}"] = cum_points.loc[t_2, f"A_{feat}{suffix}"]
        return cum_points
    

    def compute_output(self):
        dates = list(self._matches.str_date.unique())
        i = 0
        df = None
        for date, g in self._matches.groupby(by='str_date'):
            if i > 0:
                date_rank = dates[i-1]
                features = self._ranking_by_date[date_rank]
                cols = ['P', 'H_P', 'A_P', 'G', 'H_G', 'A_G', 'GA', 'H_GA', 'A_GA', 'DIFF',
                    'H_DIFF', 'A_DIFF', 'P_rank', 'H_P_rank', 'A_P_rank', 'G_rank',
                    'H_G_rank', 'A_G_rank', 'GA_rank', 'H_GA_rank', 'A_GA_rank',
                    '3M_P', 'H_3M_P', 'A_3M_P', '3M_G', 'H_3M_G', 'A_3M_G', '3M_GA',
                    'H_3M_GA', 'A_3M_GA', '3M_DIFF', 'H_3M_DIFF', 'A_3M_DIFF',
                    '3M_P_rank', '3M_H_P_rank', '3M_A_P_rank', '3M_G_rank', '3M_H_G_rank',
                    '3M_A_G_rank', '3M_GA_rank', '3M_H_GA_rank', '3M_A_GA_rank', 
                    '3M_P_coeff_rank', '3M_A_P_coeff_rank', '3M_H_P_coeff_rank', '3M_G_coeff_rank', 
                    '3M_H_G_coeff_rank', '3M_A_G_coeff_rank',
                    '3M_GA_coeff_rank','3M_A_GA_coeff_rank','3M_H_GA_coeff_rank']
                g = g[['date', 'str_date', 'country', 'league', 'season', '1_team', '2_team',  'score_ft_1', 'score_ft_2', '1_pts', '2_pts', '1_n', '1_Hn', 
                    '1_An', '2_n', '2_Hn', '2_An',
                    'bet365_1X2 Full Time_outcome_1_closing_value' ,'bet365_1X2 Full Time_outcome_2_closing_value',
                    'bet365_1X2 Full Time_outcome_3_closing_value']]
                
                g = g.merge(features[cols], left_on='1_team', right_index=True)
                g = g.merge(features[cols], left_on='2_team', right_index=True, suffixes=("_1", "_2"))

                df = pd.concat([df, g])

            else: df = g[['date', 'str_date', 'country', 'league', 'season', '1_team', '2_team', 'score_ft_1', 'score_ft_2', '1_pts', '2_pts']]
            i += 1

            self.output = df

        
    
    @staticmethod
    def cpt_points(g1, g2):
        if g1 > g2:
            return 3
        elif g1 < g2:
            return 0
        else:
            return 1

    @staticmethod
    def cpt_winner(g1, g2):
        if g1 > g2:
            return 1
        elif g1 < g2:
            return 3
        else:
            return 2

    @staticmethod
    def get_league_path(league, country, season):
        name = f"{league}_{country}_{season}".replace(' ', '_').replace('/', '_')
        return f"{LEAGUE_FOLDER}/{name}.csv"


if __name__ == '__main__':
    df = pd.read_csv(RAW_FILE)
    df = df[['date', 'country', 'league', 'season', '1_team', '2_team', 'score_ft_1', 'score_ft_2', 
    'bet365_1X2 Full Time_outcome_1_closing_value', 'bet365_1X2 Full Time_outcome_2_closing_value', 
    'bet365_1X2 Full Time_outcome_3_closing_value', 'Team1-note', 'Team2-note']].sort_values(by='date', ascending=True)
    priority = pd.read_csv(PRIORITY_LEAGUES_FILE, sep=';')
    priority = priority[priority.Prioritaire == 'OUI'][['Country', 'League']].apply(tuple, 1)
    df['country/league'] = df[['country', 'league']].apply(tuple, 1)
    df = df[df['country/league'].isin(priority)]
    df['league_instance'] = df[['league', 'country', 'season']].apply(tuple, 1)
    leagues = df.groupby('league_instance')
    print(f"N_LEAGUES: {len(leagues)}")
    for i, g in leagues:
        league_path = League.get_league_path(i[0], i[1], i[2])
        print(league_path, len(g))
        
        if len(g) > 100:
            if os.path.exists(league_path):
                continue
            else:
                _league = League(g, path=league_path)
                pickle.dump({
                    "matches": _league._matches,
                    "rankings": _league._ranking_by_date
                }, open(f"leagues/{_league.name}.pkl", "wb+"))

            