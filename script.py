import pandas as pd

db_prod = pd.read_csv('legacy/db_prod.csv').rename(
    columns={
        'bet365_1X2 Full Time_outcome_1_closing_value': 'bet365_1', 
        'bet365_1X2 Full Time_outcome_2_closing_value': 'bet365_2', 
        'bet365_1X2 Full Time_outcome_3_closing_value': 'bet365_3',
        'bet365_1X2 1st Half_outcome_1_closing_value': 'bet365_ht_1',
        'bet365_1X2 1st Half_outcome_2_closing_value': 'bet365_ht_2',
        'bet365_1X2 1st Half_outcome_3_closing_value': 'bet365_ht_3', 
        'bet365_Over/Under Full Time 2.50_outcome_1_closing_value': 'bet365_O',
        'bet365_Over/Under Full Time 2.50_outcome_2_closing_value': 'bet365_U'
    }
)
db_prod["date"] = pd.to_datetime(db_prod.date)
db_prod['str_date'] = db_prod.date.dt.to_period('D')
db_prod["id"] = db_prod[['str_date', '1_team', '2_team', 'country', 'league']].apply(tuple, axis=1)
db = pd.read_csv('db_prod_updated.csv')
db["date"] = pd.to_datetime(db.date)
db['str_date'] = db.date.dt.to_period('D')
db["id"] = db[['str_date', '1_team', '2_team', 'country', 'league']].apply(tuple, axis=1)
del db['bet365_1']
del db['bet365_2']
del db['bet365_3']
db = db.merge(db_prod[['id', 'bet365_1', 'bet365_2', 'bet365_3', 'bet365_ht_1', 'bet365_ht_2', 'bet365_ht_3', 'bet365_O', 'bet365_U']],
    how='left', on='id')

db.to_csv('db_final.csv', index=False)


exit()
countries = pd.read_csv('countries.csv')
leagues = pd.read_csv('trackable_leagues_final.csv', sep=";")
countries["alpha-2"] = countries["alpha-2"].str.lower()
leagues = leagues.merge(countries[["name", "alpha-2"]], left_on='country', right_on='alpha-2').rename(
    columns={
        "country": "s",
        "name_x": "league",
        "name_y": "country"
    }
)
leagues["season"] = "21-22"
leagues = leagues[leagues.track==1]
leagues[["country", "league", "season", "id"]].to_csv('tracked_leagues_final.csv', index=False)