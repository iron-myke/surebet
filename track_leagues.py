from distutils.command.upload import upload
import json
import os
import pickle
from datetime import datetime
from time import time
from dateutil.relativedelta import relativedelta
from pytz import timezone
import pandas as pd
import numpy as np
import gspread
import traceback

from betsapi import *
from strategy import Strategy
from league import League

FOLDER_ID = '1Poof51tpwDeuGU5zdMStaAIw8hsZJzoW'

def update_league(league):
    league_path = League.get_league_path(league.league, league.country, league.season)
    if not os.path.exists(league_path):
        matches = get_league_history_with_odds(league.id)
        if len(matches) == 0:
            return None, None
        matches["country"] = league.country
        matches["season"] = league.season
        matches["league"] = league.league
        _league = League(matches, path=league_path)
        pickle.dump({
                    "matches": _league._matches,
                    "rankings": _league._ranking_by_date
                }, open(league_path.replace('.csv', '.pkl'), "wb+"))
        last_date = list(_league._ranking_by_date.keys())[-1]
        last_ranking = _league._ranking_by_date[last_date]
    else:
        _matches = pd.read_csv(league_path)
        last_date = _matches.date.max()
        new_matches = get_league_history_with_odds(league.id, last_date)
        new_matches = new_matches[new_matches.date > last_date]
        new_matches["country"] = league.country
        new_matches["season"] = league.season
        new_matches["league"] = league.league
        print(f'{len(new_matches)} new ended matches.')
        _league_dict = pickle.load(open(league_path.replace('csv', 'pkl'), 'rb+'))
        if len(new_matches) > 0:
            print('Updating league rankings...')
            _league = League(new_matches, path=league_path, from_dict=_league_dict)
            pickle.dump({
                "matches": _league._matches,
                "rankings": _league._ranking_by_date
            }, open(f"leagues/{_league.name}.pkl", "wb+"))
            last_date = list(_league._ranking_by_date.keys())[-1]
            last_ranking = _league._ranking_by_date[last_date]
        else:
            last_key = list(_league_dict["rankings"].keys())[-1]
            last_ranking = _league_dict["rankings"][last_key]
    
    return last_date, last_ranking

def get_upcoming_matches_with_last_features(league, last_features):
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
    
    limit_ts = datetime.now() + relativedelta(days=7)
    matches = get_upcoming_league_matches(league.id, limit_ts.strftime("%Y-%m-%d %H:%M:%S"))
    print(f'{len(matches)} upcoming matches with odds')
    if len(matches) == 0:
        return None
    matches["season"] = league.season
    matches["league"] = league.league
    matches["country"] = league.country
    matches = matches.merge(last_features[cols], left_on='1_team', right_index=True)
    matches = matches.merge(last_features[cols], left_on='2_team', right_index=True, suffixes=("_1", "_2"))
    return matches

def process_leagues(leagues_df):
    print("Updating tracked leagues...")
    upcoming_matches = pd.DataFrame()
    for i, r in leagues_df.iterrows():
        try:
            print(f"LEAGUE: {r.league}, {r.country}, {r.season}")
            last_date, last_features = update_league(r)
            print("Last update date: ", last_date)
            if last_date is None:
                print('No endend matches, no features')
                print('')
                continue
            print(last_features)
            _upcoming_matches = get_upcoming_matches_with_last_features(r, last_features)
            if _upcoming_matches is not None:
                upcoming_matches = pd.concat([upcoming_matches, _upcoming_matches]).reset_index(drop=True)
            print()
        except:
            print('ERROR')
            traceback.print_exc()
    print("Done.")
    print()

    print(f"All upcoming matches: {len(upcoming_matches)}")
    STRATEGIES = os.listdir('selected_strategies')
    print(f"{len(STRATEGIES)} tracked strategies.")
    print('Uploading a Google sheet to Google Drive with a worksheet for each tracked strategy...')
    
    upload_df = pd.DataFrame()
    print(upcoming_matches.columns)
    if len(upcoming_matches) > 0:
        for i, s in enumerate(os.listdir('selected_strategies')):
            strategy = Strategy.load_strategy_from_file(f'selected_strategies/{s}')
            print(i + 1, '/', len(STRATEGIES))
            print(f"Strategy: {s}")
            _matches = Strategy.filter_matches(upcoming_matches, strategy)
            if len(_matches) > 0:
                print(f'{len(_matches)} matches selected')
                _matches.loc[:, "bet"] = strategy["result"]
                _matches.loc[:, "strategy"] = s
                _matches = _matches.replace(np.NaN, '')[["date", "country", "league", "season", "1_team", "2_team", 'bet', 'strategy', 'bet365_1', 'bet365_2','bet365_3', 'bet365_U', 'bet365_O', 'bet365_ht_1', 'bet365_ht_2', 'bet365_ht_3']].reset_index(drop=True)
                upload_df = pd.concat([upload_df, _matches])
            print()
    if len(upload_df) > 0:
        upload_df = upload_df.reset_index(drop=True)
        upload_df.loc[:, "upload_ts"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
        upload_df = upload_df.sort_values(by=['date', 'league', '1_team'])
    else:
        print("No selected matches")
    return upload_df

if __name__ == '__main__':
    tracked_leagues_arjel = pd.read_csv('tracked_leagues_final.csv')
    tracked_leagues_not_arjel = pd.read_csv('tracked_leagues_not_arjel.csv')

    upload_df = process_leagues(tracked_leagues_arjel)
    upload_df_not_arjel = process_leagues(tracked_leagues_not_arjel)

    gc = gspread.service_account()
    paris = timezone('Europe/Paris')
    ts = datetime.now(tz=paris).strftime("%y-%m-%d %H:%M")
    sheet = gc.create(f"surebet - {ts}", folder_id=FOLDER_ID)
    sheet.share('mikhail.bsa@gmail.com', perm_type='user', role='writer')
    sheet.share('cpottier@gmail.com', perm_type='user', role='writer')
    sheet.share('jeanlouisrossi@gmail.com', perm_type='user', role='writer')
    worksheet = sheet.get_worksheet(0)
    worksheet.update_title("ARJEL")
    #itle="matches", rows=upload_df.shape[0] + 10, cols=upload_df.shape[1])
    worksheet.update([upload_df.columns.values.tolist()] + upload_df.values.tolist())
    print()
    worksheet2 = sheet.add_worksheet(title="NOT ARJEL", rows=500, cols=30)
    worksheet2.update([upload_df_not_arjel.columns.values.tolist()] + upload_df_not_arjel.values.tolist())







