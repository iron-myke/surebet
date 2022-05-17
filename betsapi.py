import logging
from matplotlib.pyplot import get
import requests
import json
import os
import pickle
import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
from league import League
import traceback
from tqdm import tqdm

API_TOKEN = "123322-dvecdQtaFfhaah"
LEAGUE_IDS = [876, 99] 
BASE_URL = 'https://api.b365api.com/'

def get_leagues(cc=None):
    leagues = []
    params = {
        "token": API_TOKEN,
        "sport_id": 1,
        "page": 1
    }
    url = BASE_URL + f"v1/league"
    r = requests.get(url, params=params)
    response = r.json()
    n_pages = response.get("pager").get("total") // response.get("pager").get("per_page") + 1
    print(response.get('pager'))
    leagues += list(map(lambda x: {'id': x.get('id'), 'name': x.get('name'), 'country': x.get('cc'), 'league_table': x.get('has_leaguetable')}, response.get('results')))

    for page_id in range(2, n_pages):
        print(page_id, n_pages)
        params["page"] = page_id
        r = requests.get(url, params=params)
        response = r.json()
        leagues += list(
            map(
                lambda x: {'id': x.get('id'), 'name': x.get('name'), 'country': x.get('cc'), 'league_table': x.get('has_leaguetable')},
                response.get('results')
            )
        )
    leagues = pd.DataFrame(leagues)
    return leagues

def get_league(league_id):
    params = {
        "token": API_TOKEN,
        "league_id": league_id,
    }
    url = BASE_URL + "v2/league/table"
    r = requests.get(url, params=params)
    response = r.json()
    start_time = int(response.get('results').get('season').get('start_time'))
    name =  response.get('results').get('season').get('name')
    dt_object = datetime.fromtimestamp(start_time)
    start_date_string = dt_object.strftime("%Y-%m-%d %H:%M:%S")
    return {"name": name , "start_date": start_date_string, "country": name.split(' ')[-1], "start_ts": start_time}

def get_match(match_id):
    params = {
            "token": API_TOKEN,
            "event_id": match_id,
        }
    url = BASE_URL + "v1/event/view"
    r = requests.get(url, params=params)
    response = r.json()
    print(response)

def get_odds(match_id, latest:bool=False, start_time=None):
    params = {
        "token": API_TOKEN,
        "event_id": match_id,
        "source": "bet365",
        "odds_market": "1,3,8"
    }
    url = BASE_URL + "v2/event/odds"
    r = requests.get(url, params=params)
    response = r.json()
    index = 0 if latest else -1
    odds_dic = {"match_id": match_id}

    for odd_type, odd_name, odd_list, odd_names_list in zip(
        [1, 3, 8], ["", "", "_ht"],
        [['home_od', 'draw_od', 'away_od'], ['under_od', "over_od"], ['home_od', 'draw_od', 'away_od']],
        [[1, 2, 3], ['U', "O"], [1, 2, 3]]):
        all_odds = response["results"]["odds"].get(f"1_{odd_type}", [])
        if len(all_odds) == 0:
            odds_dic.update({f"bet365{odd_name}_{ot_name}": None for ot, ot_name in zip(odd_list, odd_names_list)})
            continue
        if start_time:
            odds = next((x for x in all_odds if x["add_time"] < str(start_time)))
        else:
            odds = all_odds[index]
        odds_dic.update({f"bet365{odd_name}_{ot_name}": odds[ot] for ot, ot_name in zip(odd_list, odd_names_list)})
    
    return odds_dic

def get_ended_league_matches(league_id, start_date=None): 
    league = get_league(league_id)
    if start_date is None:
        start_date = league.get('start_date')
    url = BASE_URL + f"v3/events/ended"
    page_id = 1
    first_date = start_date
    matches = []
    n_pages = 1
    n_matches = 0
    print("Fetching ended matches...")
    while first_date >= start_date and page_id <= n_pages:
        params = {
            "token": API_TOKEN,
            "sport_id": 1,
            "league_id": league_id,
            "page": page_id
        }
        r = requests.get(url, params=params)
        response = r.json()
        if n_pages == 1:
            n_pages = response.get('pager', {}).get('total', 1) / response.get('pager', {}).get('per_page', 50)
        for r in response.get("results"):
            if r.get('time_status') != "3":
                continue
            first_date = datetime.fromtimestamp(int(r.get('time'))).strftime("%Y-%m-%d %H:%M:%S")
            if first_date < start_date:
                break
            match = {
                "league": r.get('league').get('name'),
                "date": first_date,
                "match_id": r.get('id'),
                "1_team": r.get('home').get('name'),
                "2_team": r.get('away').get('name'), 
                "score": r.get('ss'),
                "ts": r.get('time')
            }
            matches.append(match)
            n_matches += 1
        print(f"Batch {page_id} done, {n_matches} matches added, last_date: {first_date}")
        n_matches = 0
        page_id += 1
    matches = pd.DataFrame(matches)
    if len(matches) == 0:
        return pd.DataFrame()
    matches = matches[matches.score.notna()]
    matches["score_ft_1"] = matches.score.apply(lambda x: int(x.split('-')[0]))
    matches["score_ft_2"] = matches.score.apply(lambda x: int(x.split('-')[1]))
    print("Done.")
    return matches


def get_league_history_with_odds(league_id, start_date=None):
    matches = get_ended_league_matches(league_id, start_date)
    print("Fetching odds...")
    if len(matches) == 0:
        return pd.DataFrame()
    odds = []
    for i in tqdm(range(len(matches))):
        r = matches.iloc[i]
        try: 
            _odds = get_odds(r.match_id, start_time=r.ts)
            odds.append(_odds)
        except Exception:
            print(f'No odds for match {r.match_id}, {r["1_team"]} - {r["2_team"]}, {r.date}')
            #traceback.print_exc()
    odds = pd.DataFrame(odds)
    if len(odds) > 0:
        matches = matches.merge(odds, how='left', on='match_id')
    print("Done")
    return matches

def get_upcoming_league_matches(league_id, limit_ts=None): 
    league = get_league(league_id)  
    url = BASE_URL + f"v3/events/upcoming"
    page_id = 1
    matches = []
    params = {
        "token": API_TOKEN,
        "sport_id": 1,
        "league_id": league_id,
        "page": page_id,
        "day": "TODAY" #"20220417" #datetime.now().strftime('%y%m%d')
    }
    r = requests.get(url, params=params)
    response = r.json()
    for r in response.get("results"):
        try: 
            odds = get_odds(match_id=r.get('id'), latest=True)
            date = datetime.fromtimestamp(int(r.get('time'))).strftime("%Y-%m-%d %H:%M:%S")
            if (limit_ts is not None) and (date > limit_ts):
                break 
            match = dict({
                "league": league.get('name'),
                "date": date,
                "ts": r.get("time"),
                "match_id": r.get('id'),
                "1_team": r.get('home').get('name'),
                "2_team": r.get('away').get('name')
            }, **odds)
            matches.append(match)
        except:
            break
    matches = pd.DataFrame(matches)
    return matches

def get_league_seasons(league_id):
    league = get_league(league_id)  
    start_date = datetime.fromtimestamp(league["start_ts"]) + relativedelta(days=-15)
    limit_date = start_date + relativedelta(years=-5)
    matches = get_league_history_with_odds(league_id, start_date=limit_date.strftime("%Y-%m-%d %H:%M:%S"))
    matches.date = pd.to_datetime(matches.date)
    cut_date = start_date
    year = 20
    df = pd.DataFrame()
    while cut_date > limit_date:
        new_cut_date = cut_date + relativedelta(years=-1)
        season = matches[(matches.date < cut_date) & (matches.date > new_cut_date)].drop_duplicates(["1_team", "2_team", "date"])
        season["season"] = f"{year}-{year+1}"
        matches = matches[~matches.match_id.isin(season.match_id)]
        print(cut_date, new_cut_date, len(season))
        print(season)
        print("x")
        cut_date = new_cut_date
        year += -1
        df = pd.concat([df, season])
    df = df.reset_index(drop=True)
    return df

if __name__ == '__main__':
    tracked_leagues = pd.read_csv('tracked_leagues_final.csv')
    for i, r in tracked_leagues.iterrows():
        league_path = League.get_league_path(r.league, r.country, "20-21")
        if not os.path.exists(league_path):
            print("Processing", r.league)
            _matches = get_league_seasons(r.id) #pd.read_csv('sisi')
            _matches["country"] = r.country
            _matches.to_csv('sisi.csv')
            os.system('python3 league.py -f sisi.csv')
            #print(season)
        #print(get_match(3951412))

    #print(get_odds(3951412, True, 1649604600))
    #print(get_odds(3951412, True))
    #print(get_odds(3951412, False))

    #print(leagues)
    #leagues = leagues[leagues.league_table == 1].reset_index(drop=True)
    #leagues = leagues.dropna(subset=['country'])
    #leagues = leagues[~leagues.name.str.contains('women', False)]
    #leagues = leagues[~leagues.name.str.contains('cup', False)]
    #leagues = leagues[~leagues.name.str.contains('play-offs', False)]
    #leagues = leagues[~leagues.name.str.contains('relegation', False)]
    #leagues = leagues[~leagues.name.str.contains('playoff', False)]
    #leagues = leagues[~leagues.name.str.contains('group', False)]
    #leagues = leagues[~leagues.name.str.contains('copa', False)]

    #print(leagues)
    #leagues.to_csv('trackable_leagues.csv', index=False)