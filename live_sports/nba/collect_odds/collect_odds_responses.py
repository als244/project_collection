import requests
import json
import pickle
import datetime
import time
import os
import pandas as pd
import numpy as np
from datetime import date

# An api key is emailed to you when you sign up to a plan
# Get a free API key at https://api.the-odds-api.com/
API_KEY = '31e4f615972c37f63ab734fa92a2de10'
SPORT = "basketball_nba"
REGIONS = 'us' # uk | us | eu | au. Multiple can be specified if comma delimited

MARKETS = 'h2h' # h2h | spreads | totals. Multiple can be specified if comma delimited

ODDS_FORMAT = 'decimal' # decimal | american

DATE_FORMAT = 'unix' # iso | unix

LOOP_FREQ = 30.0

today = date.today()

DATE_STR = today.strftime("%Y%m%d")
ODDS_PREF = "/mnt/storage/data/live_sports/nba/odds/" + DATE_STR

if not os.path.exists(ODDS_PREF):
    os.makedirs(ODDS_PREF)

if not os.path.exists(ODDS_PREF + "/responses"):
    os.makedirs(ODDS_PREF + "/responses")

start_time = time.time()


### CAN LEAVE THIS RUNNING TO KEEP GENERATING UPDATED LIVE ODDS

## BE CAREFUL OF API LIMITS...

while True:
    ts = time.time()
    time_human = datetime.datetime.fromtimestamp(ts).strftime('%H_%M_%S')
    dt = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    print(dt)

    odds_response = requests.get(
        f'https://api.the-odds-api.com/v4/sports/{SPORT}/odds/',
        params={
            'api_key': API_KEY,
            'regions': REGIONS,
            'markets': MARKETS,
            'oddsFormat': ODDS_FORMAT,
            'dateFormat': DATE_FORMAT,
        }
    )

    if odds_response.status_code != 200:
        print(f'Failed to get odds: status_code {odds_response.status_code}, response body {odds_response.text}')

    else:
        odds_json = odds_response.json()
        print('Number of events:', len(odds_json))
        print(odds_json)

        # Check the usage quota
        print('Remaining requests', odds_response.headers['x-requests-remaining'])
        print('Used requests', odds_response.headers['x-requests-used'])


        pickle.dump(odds_response, open(ODDS_PREF + "/" + "responses" + "/" + time_human + ".pickle", "wb"))
        time.sleep(LOOP_FREQ - ((time.time() - start_time) % LOOP_FREQ))