import json
import numpy as np
import os
import pandas as pd

CONFIG_FILEPATH = "config.json"
CONFIG = json.load(open(CONFIG_FILEPATH, 'r'))
LEAGUE_FOLDER = CONFIG.get("LEAGUE_FOLDER")
RAW_FILE = CONFIG.get("RAW_FILE")
PRIORITY_LEAGUES_FILE = CONFIG.get("PRIORITY_LEAGUES_FILE")
GA_COEFF_FILE = CONFIG.get("GA_COEFF_FILE")
POINTS_COEFF_FILE = CONFIG.get("POINTS_COEFF_FILE")

class League: 

    def __init__(self, df, path):
        self.preprocessing(df)
        self.compute_ranking_by_date()
        #self.compute_M3_ranking_by_date()
        #self.compute_M3_coeff_ranking_by_date()
        #self.compute_output()
        #self.output.to_csv(path)


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
        print("Teams: ", self._teams, len(self._teams))
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
        self._matches_by_date = self._matches.groupby('str_date')
        features = [x + y for x in['H_', 'A_', '',] for y in ['P', 'G', 'GA', '3M_P', '3M_G', '3M_GA']]
        cum_points = pd.DataFrame.from_dict({t: {f:0 for f in features} for t in self._teams}, orient="index")
        ranking_by_date = {}
        ranking_by_date[0] = cum_points
        last_date = 0
        for i, g in self._matches_by_date:
            cum_points = ranking_by_date[last_date].copy()
            for i_m, m in g.iterrows():
                t_1, t_2 = m["1_team"], m["2_team"]
                cum_points.loc[t_1, ["P", "H_P"]] += m["1_pts"] 
                cum_points.loc[t_1, ["G", "H_G"]] += m["score_ft_1"] 
                cum_points.loc[t_1, ["GA", "H_GA"]] += m["score_ft_2"] 
            
                cum_points.loc[t_2, ["P", "A_P"]] += m["2_pts"] 
                cum_points.loc[t_2, ["G", "A_G"]] += m["score_ft_2"] 
                cum_points.loc[t_2, ["GA", "A_GA"]] += m["score_ft_1"] 
                
                if m["1_n"] >= 3:
                    _M3_match = self._matches[((self._matches['1_team'] == t_1) & (self._matches['1_n'] == m["1_n"] - 3)) | ((self._matches['2_team'] == t_1) & (self._matches['2_n'] == m["1_n"] - 3))]
                    _3M_cum_points = ranking_by_date[_M3_match.str_date.iloc[0]]
                    cum_points.loc[t_1, "3M_P"] = cum_points.loc[t_1, "P"] - _3M_cum_points.loc[t_1, "P"]
                    cum_points.loc[t_1, "3M_G"] = cum_points.loc[t_1, "G"] - _3M_cum_points.loc[t_1, "G"]
                    cum_points.loc[t_1, "3M_GA"] = cum_points.loc[t_1, "GA"] - _3M_cum_points.loc[t_1, "GA"]
                else:
                    cum_points.loc[t_1, "3M_P"] = cum_points.loc[t_1, "P"]
                    cum_points.loc[t_1, "3M_G"] = cum_points.loc[t_1, "G"]
                    cum_points.loc[t_1, "3M_GA"] = cum_points.loc[t_1, "GA"]

                if m["1_Hn"] >= 3:
                    _M3_home_match = self._matches[((self._matches['1_team'] == t_1) & (self._matches['1_Hn'] == m["1_Hn"] - 3))]
                    _3M_cum_points = ranking_by_date[_M3_home_match.str_date.iloc[0]]
                    cum_points.loc[t_1, "H_3M_P"] = cum_points.loc[t_1, "H_P"] - _3M_cum_points.loc[t_1, "H_P"]
                    cum_points.loc[t_1, "H_3M_G"] = cum_points.loc[t_1, "H_G"] - _3M_cum_points.loc[t_1, "H_G"]
                    cum_points.loc[t_1, "H_3M_GA"] = cum_points.loc[t_1, "H_GA"] - _3M_cum_points.loc[t_1, "H_GA"]
                else:
                    cum_points.loc[t_1, "H_3M_P"] = cum_points.loc[t_1, "H_P"]
                    cum_points.loc[t_1, "H_3M_G"] = cum_points.loc[t_1, "H_G"]
                    cum_points.loc[t_1, "H_3M_GA"] = cum_points.loc[t_1, "H_GA"]

                if m["2_n"] >= 3:
                    _M3_match = self._matches[((self._matches['1_team'] == t_2) & (self._matches["1_n"] == m["2_n"] - 3)) | ((self._matches['2_team'] == t_2) & (self._matches['2_n'] == m["2_n"] - 3))]
                    _3M_cum_points = ranking_by_date[_M3_match.str_date.iloc[0]]
                    cum_points.loc[t_2, "3M_P"] = cum_points.loc[t_2, "P"] - _3M_cum_points.loc[t_2, "P"]
                    cum_points.loc[t_2, "3M_G"] = cum_points.loc[t_2, "G"] - _3M_cum_points.loc[t_2, "G"]
                    cum_points.loc[t_2, "3M_GA"] = cum_points.loc[t_2, "GA"] - _3M_cum_points.loc[t_2, "GA"]
                else:
                    cum_points.loc[t_2, "3M_P"] = cum_points.loc[t_2, "P"]
                    cum_points.loc[t_2, "3M_G"] = cum_points.loc[t_2, "G"]
                    cum_points.loc[t_2, "3M_GA"] = cum_points.loc[t_2, "GA"]

                if m["2_An"] >= 3:
                    _M3_away_match = self._matches[((self._matches['2_team'] == t_2) & (self._matches["2_An"] == m["2_An"] - 3))]
                    _3M_cum_points = ranking_by_date[_M3_away_match.str_date.iloc[0]]    
                    cum_points.loc[t_2, "A_3M_P"] = cum_points.loc[t_2, "A_P"] - _3M_cum_points.loc[t_2, "A_P"]
                    cum_points.loc[t_2, "A_3M_G"] = cum_points.loc[t_2, "A_G"] - _3M_cum_points.loc[t_2, "A_G"]
                    cum_points.loc[t_2, "A_3M_GA"] = cum_points.loc[t_2, "A_GA"] - _3M_cum_points.loc[t_2, "A_GA"]
                else:
                    cum_points.loc[t_2, "A_3M_P"] = cum_points.loc[t_2, "A_P"]
                    cum_points.loc[t_2, "A_3M_G"] = cum_points.loc[t_2, "A_G"]
                    cum_points.loc[t_2, "A_3M_GA"] = cum_points.loc[t_2, "A_GA"]

            cum_points["DIFF"] = cum_points.G - cum_points.GA
            cum_points["H_DIFF"] = cum_points.H_G - cum_points.H_GA
            cum_points["A_DIFF"] = cum_points.A_G - cum_points.A_GA
            cum_points["H_3M_DIFF"] = cum_points.H_3M_G - cum_points.H_3M_GA
            cum_points["3M_DIFF"] = cum_points["3M_G"] - cum_points["3M_GA"]
            cum_points["A_3M_DIFF"] = cum_points.A_3M_G - cum_points.A_3M_GA

            l = [("P_rank", ["P", "DIFF", "G"], False), ("H_P_rank", ["H_P", "H_DIFF", "H_G"], False), 
                ("A_P_rank", ["A_P", "A_DIFF", "A_G"], False), ("G_rank", ["G"], False), 
                ("H_G_rank", ["H_G"], False), ("A_G_rank", ["A_G"], False), ("GA_rank", ["GA"], True), 
                ("H_GA_rank", ["H_GA"], True), ("A_GA_rank", ["A_GA"], True), 
                ("3M_P_rank", ["3M_P", "3M_DIFF", "3M_G"], False), ("3M_H_P_rank", ["H_3M_P", "H_3M_DIFF", "H_3M_G"], False), 
                ("3M_A_P_rank", ["A_3M_P", "A_3M_DIFF", "A_3M_G"], False), ("3M_G_rank", ["3M_G"], False), 
                ("3M_A_G_rank", ["A_3M_G"], False), ("3M_A_G_rank", ["A_3M_G"], False), ("3M_GA_rank", ["3M_GA"], True), 
                ("3M_H_GA_rank", ["H_3M_GA"], True), ("3M_A_GA_rank", ["A_3M_GA"], True)]
            
            for r, args, order in l:
                cum_points[r] = cum_points[args].apply(tuple,axis=1).rank(method='min',ascending=order).astype(int)
            
            cum_points = cum_points.sort_values(by='3M_P_rank')
            ranking_by_date[i] = cum_points
        self._ranking_by_date = ranking_by_date
        print(self._ranking_by_date)


    def compute_M3_ranking_by_date(self):
        print('Computing Last 3 matches ranking by Date...')
        n_matches = np.zeros(self.n_teams)
        M3_ranking_by_date = {}
        k_old = None
        g_cols = ["P", "DIFF", "G", "GA"]
        a_cols = ["A_P", "A_DIFF", "A_G", "A_GA"]
        h_cols = ["H_P", "H_DIFF", "H_G", "H_GA"]
        for k, v in self._ranking_by_date.items():
            _v = v.copy()
            for i, t in enumerate(self._teams):
                _df = self._matches[(self._matches.str_date == k) & ((self._matches['1_team'] == t) | (self._matches['2_team'] == t))]
                if len(_df) > 0:
                    n = _df['1_n'].iloc[0] if _df['1_team'].iloc[0] == t else _df['2_n'].iloc[0]
                    n_H = _df['1_Hn'].iloc[0] if _df['1_team'].iloc[0] == t else _df['2_Hn'].iloc[0] - 1
                    n_A = (_df['1_An'].iloc[0] - 1) if _df['1_team'].iloc[0] == t else _df['2_An'].iloc[0]
                    n_matches[i] += 1
                    
                    if(n >= 3):
                        match = self._matches[((self._matches['1_team'] == t) & (self._matches['1_n'] == n - 3)) | ((self._matches['2_team'] == t) & (self._matches['2_n'] == n - 3))]
                        _df_3 = self._ranking_by_date[match.str_date.iloc[0]]
                        _v.loc[t, g_cols] = _v.loc[t, g_cols] - _df_3.loc[t, g_cols]
                    elif k_old is not None:
                        _v.loc[t, g_cols] = _v.loc[t, g_cols]

                    if (_df['1_team'].iloc[0] == t) and n_H >= 3:
                        match = self._matches[((self._matches['1_team'] == t) & (self._matches['1_Hn'] == n_H - 3))]
                        _df_3 = self._ranking_by_date[match.str_date.iloc[0]]
                        _v.loc[t, h_cols] = _v.loc[t, h_cols] - _df_3.loc[t, h_cols]
                    elif k_old is not None:
                        _v.loc[t, h_cols] = M3_ranking_by_date[k_old].loc[t, h_cols]
                        
                    if (_df['2_team'].iloc[0] == t) and (n_A >= 3):
                        match = self._matches[((self._matches['2_team'] == t) & (self._matches['2_An'] == n_A - 3))]
                        _df_3 = self._ranking_by_date[match.str_date.iloc[0]]
                        _v.loc[t, a_cols] = _v.loc[t, a_cols] - _df_3.loc[t, a_cols]
                    elif k_old is not None:
                        _v.loc[t, a_cols] = M3_ranking_by_date[k_old].loc[t, a_cols]
                    
                elif k_old is not None:
                    _v.loc[t, :] = M3_ranking_by_date[k_old].loc[t, :]

                    
            _v = _v.replace({np.nan: 0})
            _v["P_rank"] = _v[["P", "DIFF", "G"]].apply(tuple,axis=1).rank(method='min',ascending=False).astype(int)
            _v["G_rank"] = _v[["G"]].rank(method='min',ascending=False).astype(int)
            _v["GA_rank"] = _v[["GA"]].rank(method='min',ascending=True).astype(int)
            
            _v["H_P_rank"] = _v[["H_P", "H_DIFF", "H_G"]].apply(tuple,axis=1).rank(method='min',ascending=False).astype(int)
            _v["H_G_rank"] = _v[["H_G"]].rank(method='min',ascending=False).astype(int)
            _v["H_GA_rank"] =  _v[["H_GA"]].rank(method='min',ascending=True).astype(int)
            
            _v["A_P_rank"] = _v[["A_P", "A_DIFF", "A_G"]].apply(tuple,axis=1).rank(method='min',ascending=False).astype(int)
            _v["A_G_rank"] = _v[["A_G"]].rank(method='min',ascending=False).astype(int)
            _v["A_GA_rank"] =  _v[["A_GA"]].rank(method='min',ascending=True).astype(int)
            _v = _v.sort_values(by='P_rank')
            
            M3_ranking_by_date[k] = _v
            k_old = k
        self._M3_ranking_by_date = M3_ranking_by_date


    def compute_M3_coeff_ranking_by_date(self):
        print('Computing Last 3M Weighted Ranking by Date...')
        matches_by_date = self._matches.groupby('str_date')
        dates = list(self._matches.str_date.unique())
        i = 0
        df = None
        for date, g in matches_by_date:
            if i > 0:
                date_rank = dates[i-1]
                features2 = self._M3_ranking_by_date[date_rank]
                features2 = features2.rename(columns={c:"3M_" + c for c in features2.columns})
                cols_2 = ['3M_P', '3M_H_P', '3M_A_P', '3M_G', '3M_H_G', '3M_A_G', '3M_GA',
                '3M_H_GA', '3M_A_GA', '3M_DIFF', '3M_H_DIFF', '3M_A_DIFF', '3M_N_M',
                '3M_P_rank', '3M_H_P_rank', '3M_A_P_rank', '3M_G_rank', '3M_H_G_rank',
                '3M_A_G_rank', '3M_GA_rank', '3M_H_GA_rank', '3M_A_GA_rank']
                g = g[['date', 'str_date', 'country', 'league', 'season', '1_team', '2_team', 'score_ft_1', 'score_ft_2', '1_pts', '2_pts']]
                g = g.merge(features2[cols_2], left_on='1_team', right_index=True)
                g = g.merge(features2[cols_2], left_on='2_team', right_index=True, suffixes=("_1", "_2"))
                df = pd.concat([df, g])
            else: 
                df = g[['date', 'str_date', 'country', 'league', 'season', '1_team', '2_team', 'score_ft_1', 'score_ft_2', '1_pts', '2_pts']]      
            i += 1

        coeffs = pd.read_csv(POINTS_COEFF_FILE, sep=';')
        coeffs_gA =  pd.read_csv(GA_COEFF_FILE, sep=';')
        dic = {r.opponent_rank: r.coeff for i, r in coeffs.iterrows()}
        dic2 = {r.opponent_rank: float(r.coeff.replace(',', '.')) for i, r in coeffs_gA.iterrows()}
        dic[0.0] = 1.0
        dic2[0.0] = 1.0

        _l = [
            ('1_pts_coeff', '3M_A_P_rank_2', '1_pts', dic), ('2_pts_coeff', '3M_P_rank_1', '2_pts', dic),
            ('1_pts_coeff_H', '3M_A_P_rank_2', '1_pts', dic), ('2_pts_coeff_A', '3M_H_P_rank_1', '2_pts', dic),
            ('1_G_coeff', '3M_GA_rank_2', 'score_ft_1', dic), ('2_G_coeff', '3M_GA_rank_1', 'score_ft_2', dic),
            ('1_G_coeff_H', '3M_GA_rank_2', 'score_ft_1', dic), ('2_G_coeff_A', '3M_H_GA_rank_1', 'score_ft_2', dic),
            ('1_GA_coeff', '3M_G_rank_2', 'score_ft_2', dic2), ('2_GA_coeff', '3M_G_rank_1', 'score_ft_1', dic2),
            ('1_GA_coeff_H', '3M_A_G_rank_2', 'score_ft_2', dic2), ('2_GA_coeff_A', '3M_H_G_rank_1', 'score_ft_1', dic2)
        ]

        for i, j, k, d in _l:
            df[i] = df.apply(lambda x: (d[x[j]] * x[k]) if not pd.isna(x[j]) else x[k], axis=1)

        matches_by_date = df.groupby('str_date')
        cum_points = np.zeros((self.n_teams, 15))
        cum_points_list_coeff = {}
        count = 0
        for i, g in matches_by_date:
            cum_points = cum_points.copy()
            for j, t in enumerate(self._teams):
                home_points = g.loc[(g['1_team'] == t), '1_pts_coeff_H'].sum()
                away_points = g.loc[(g['2_team'] == t), '2_pts_coeff_A'].sum()
                home_goals = g.loc[(g['1_team'] == t), '1_G_coeff_H'].sum()
                away_goals = g.loc[(g['2_team'] == t), '2_G_coeff_A'].sum()
                home_ga = g.loc[(g['1_team'] == t), '1_GA_coeff_H'].sum()
                away_ga = g.loc[(g['2_team'] == t), '2_GA_coeff_A'].sum()

                cum_points[j][0] = cum_points[j][0] + home_points + away_points
                cum_points[j][1] = cum_points[j][1] + home_points
                cum_points[j][2] = cum_points[j][2] + away_points

                cum_points[j][3] = cum_points[j][3] + home_goals +  away_goals
                cum_points[j][4] = cum_points[j][4] + home_goals
                cum_points[j][5] = cum_points[j][5] + away_goals
                
                cum_points[j][6] = cum_points[j][6] + home_ga +  away_ga
                cum_points[j][7] = cum_points[j][7] + home_ga
                cum_points[j][8] = cum_points[j][8] + away_ga

                if (t in g['1_team'].values):
                    cum_points[j][13] += 1
                elif (t in g['2_team'].values):
                    cum_points[j][14] += 1
            
            cum_points[:, 9] =  cum_points[:, 3] -  cum_points[:, 6]
            cum_points[:, 10] = cum_points[:, 4] -  cum_points[:, 7]
            cum_points[:, 11] = cum_points[:, 5] -  cum_points[:, 8]
            cum_points[:, 12] = cum_points[:, 13] + cum_points[:, 14]

            _df = pd.DataFrame(cum_points, index=self._teams, columns=[
                "P_coeff", "H_P_coeff", "A_P_coeff", "G_coeff", "H_G_coeff", "A_G_coeff", "GA_coeff", "H_GA_coeff", "A_GA_coeff", "DIFF_coeff", "H_DIFF_coeff", "A_DIFF_coeff", "N_M_coeff", "H_N_M_coeff", "A_N_M_coeff"
            ])
            cum_points_list_coeff[i] = _df
        
        n_matches = np.zeros(self.n_teams)
        n3_list_coeff = {}
        k_old = None
        g_cols = ["P_coeff", "DIFF_coeff", "G_coeff", "GA_coeff"]
        a_cols = ["A_P_coeff", "A_DIFF_coeff", "A_G_coeff", "A_GA_coeff"]
        h_cols = ["H_P_coeff", "H_DIFF_coeff", "H_G_coeff", "H_GA_coeff"]
        for k, v in cum_points_list_coeff.items():
            _v = v.copy()
            for i, t in enumerate(self._teams):
                _df = self._matches[(self._matches.str_date == k) & ((self._matches['1_team'] == t) | (self._matches['2_team'] == t))]
                if len(_df) > 0:
                    n = _df['1_n'].iloc[0] if _df['1_team'].iloc[0] == t else _df['2_n'].iloc[0]
                    n_H = _df['1_Hn'].iloc[0] if _df['1_team'].iloc[0] == t else _df['2_Hn'].iloc[0] - 1
                    n_A = (_df['1_An'].iloc[0] - 1) if _df['1_team'].iloc[0] == t else _df['2_An'].iloc[0]
                    n_matches[i] += 1
                    
                    if(n >= 3):
                        match = self._matches[
                            ((self._matches['1_team'] == t) & (self._matches['1_n'] == n - 3)
                            ) | ((self._matches['2_team'] == t) & (self._matches['2_n'] == n - 3))]
                        _df_3 = cum_points_list_coeff[match.str_date.iloc[0]]
                        _v.loc[t, g_cols] = _v.loc[t, g_cols] - _df_3.loc[t, g_cols]
                    elif k_old is not None:
                        _v.loc[t, g_cols] = _v.loc[t, g_cols]

                    if (_df['1_team'].iloc[0] == t) and n_H >= 3:
                        match = self._matches[((self._matches['1_team'] == t) & (self._matches['1_Hn'] == n_H - 3))]
                        _df_3 = cum_points_list_coeff[match.str_date.iloc[0]]
                        _v.loc[t, h_cols] = _v.loc[t, h_cols] - _df_3.loc[t, h_cols]
                    elif k_old is not None:
                        _v.loc[t, h_cols] = n3_list_coeff[k_old].loc[t, h_cols]
                        
                    if (_df['2_team'].iloc[0] == t) and (n_A >= 3):
                        match = self._matches[((self._matches['2_team'] == t) & (self._matches['2_An'] == n_A - 3))]
                        _df_3 = cum_points_list_coeff[match.str_date.iloc[0]]
                        _v.loc[t, a_cols] = _v.loc[t, a_cols] - _df_3.loc[t, a_cols]
                    elif k_old is not None:
                        _v.loc[t, a_cols] = n3_list_coeff[k_old].loc[t, a_cols]
                    
                elif k_old is not None:
                    _v.loc[t, :] = n3_list_coeff[k_old].loc[t, :]
     
            _v = _v.replace({np.nan: 0})
            _v["P_rank_coeff"] = _v[["P_coeff", "DIFF_coeff", "G_coeff"]].apply(tuple,axis=1).rank(method='min',ascending=False).astype(int)
            _v["G_rank_coeff"] = _v[["G_coeff"]].rank(method='min',ascending=False).astype(int)
            _v["GA_rank_coeff"] = _v[["GA_coeff"]].rank(method='min',ascending=True).astype(int)
            
            _v["H_P_rank_coeff"] = _v[["H_P_coeff", "H_DIFF_coeff", "H_G_coeff"]].apply(tuple,axis=1).rank(method='min',ascending=False).astype(int)
            _v["H_G_rank_coeff"] = _v[["H_G_coeff"]].rank(method='min',ascending=False).astype(int)
            _v["H_GA_rank_coeff"] =  _v[["H_GA_coeff"]].rank(method='min',ascending=True).astype(int)
            
            _v["A_P_rank_coeff"] = _v[["A_P_coeff", "A_DIFF_coeff", "A_G_coeff"]].apply(tuple,axis=1).rank(method='min',ascending=False).astype(int)
            _v["A_G_rank_coeff"] = _v[["A_G_coeff"]].rank(method='min',ascending=False).astype(int)
            _v["A_GA_rank_coeff"] =  _v[["A_GA_coeff"]].rank(method='min',ascending=True).astype(int)
            _v = _v.sort_values(by='P_rank_coeff')
            
            n3_list_coeff[k] = _v
            k_old = k
        self._M3_coeff_ranking_by_date = n3_list_coeff


    def compute_output(self):
        dates = list(self._matches.str_date.unique())
        i = 0
        df = None
        for date, g in self._matches.groupby(by='str_date'):
            if i > 0:
                date_rank = dates[i-1]
                features = self._ranking_by_date[date_rank]
                cols = ['P', 'H_P', 'A_P', 'G', 'H_G', 'A_G', 'GA', 'H_GA', 'A_GA', 'DIFF',
                'H_DIFF', 'A_DIFF', 'N_M', 'P_rank', 'H_P_rank', 'A_P_rank', 'G_rank',
                'H_G_rank', 'A_G_rank', 'GA_rank', 'H_GA_rank', 'A_GA_rank'
                ]
                features2 = self._M3_ranking_by_date[date_rank]
                features2 = features2.rename(columns={c:"3M_" + c for c in features2.columns})
                cols_2 = ['3M_P', '3M_H_P', '3M_A_P', '3M_G', '3M_H_G', '3M_A_G', '3M_GA',
                '3M_H_GA', '3M_A_GA', '3M_DIFF', '3M_H_DIFF', '3M_A_DIFF', '3M_N_M',
                '3M_P_rank', '3M_H_P_rank', '3M_A_P_rank', '3M_G_rank', '3M_H_G_rank',
                '3M_A_G_rank', '3M_GA_rank', '3M_H_GA_rank', '3M_A_GA_rank']
                
                features3 = self._M3_coeff_ranking_by_date[date_rank]
                features3 = features3.rename(columns={c:"3M_" + c for c in features3.columns})
                cols_3 = ['3M_P_rank_coeff', '3M_A_P_rank_coeff', '3M_H_P_rank_coeff', '3M_G_rank_coeff', '3M_H_G_rank_coeff', '3M_A_G_rank_coeff',
                        '3M_GA_rank_coeff','3M_A_GA_rank_coeff','3M_H_GA_rank_coeff']

                g = g[['date', 'str_date', 'country', 'league', 'season', '1_team', '2_team',  'score_ft_1', 'score_ft_2', '1_pts', '2_pts',
                    'bet365_1X2 Full Time_outcome_1_closing_value' ,'bet365_1X2 Full Time_outcome_2_closing_value',
                    'bet365_1X2 Full Time_outcome_3_closing_value', 'bet365_1X2 1st Half_outcome_1_closing_value',
                    'bet365_1X2 1st Half_outcome_2_closing_value', 'bet365_1X2 1st Half_outcome_3_closing_value',
                    'bet365_Over/Under Full Time 2.50_outcome_1_closing_value','bet365_Over/Under Full Time 2.50_outcome_2_closing_value']]
                for f, c in [(features, cols), (features2, cols_2), (features3, cols_3)]:
                    g = g.merge(f[c], left_on='1_team', right_index=True)
                    g = g.merge(f[c], left_on='2_team', right_index=True, suffixes=("_1", "_2"))

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
        
        if len(g) > 50:
            if os.path.exists(league_path):
                continue
            else:
                _league = League(g, path=league_path)
            