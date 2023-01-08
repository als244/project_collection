import requests
import json
import pickle
import datetime
import time
import pandas as pd
import numpy as np
import os
from datetime import date

today = date.today()

SCOREBOARD_PREF = "/mnt/storage/data/live_sports/nba/stats/scoreboards"
DATE_STR = today.strftime("%Y%m%d")

scoreboard_df = pickle.load(open(SCOREBOARD_PREF + "/" + DATE_STR + ".pickle", "rb"))

game_ids = scoreboard_df["game_id"]

PLAYBYPLAY_PREF = "/mnt/storage/data/live_sports/nba/stats/playbyplay"

if not os.path.exists(PLAYBYPLAY_PREF + "/" + DATE_STR):
	os.makedirs(PLAYBYPLAY_PREF + "/" + DATE_STR)

LOOP_FREQ = 30.0

START_TIME = time.time()

### KEEP SCRIPT RUNNING DURING GAMES TO COLLECT UPDATED PLAY BY PLAY

### WOULD LIKE TO MAKE THIS STREAMING INSTEAD OF BUILDING UP NEW FRAMES AND REPEATING DATA...
while True:
	ts = time.time()
	time_human = datetime.datetime.fromtimestamp(ts).strftime('%H_%M_%S')
	dt = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

	action_attributes = ["actionNumber", "clock", "timeActual", "period", "scoreHome", "scoreAway", "teamId", "teamTricode", "actionType", "subType", "descriptor", "description", "personId", "playerName", "isFieldGoal", "shotResult", "x", "y", "area", "areaDetail", "side", "shotDistance", "shotActionNumber", "possession", "edited"]
	
	for g in game_ids:
		
		if not os.path.exists(PLAYBYPLAY_PREF + "/" + DATE_STR + "/" + str(g)):
			os.makedirs(PLAYBYPLAY_PREF + "/" + DATE_STR + "/" + str(g))

		playbyplay_req_str = "https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_" + str(g) + ".json"
		playbyplay_resp = requests.get(playbyplay_req_str)

		# happens for games that haven't started
		if playbyplay_resp.status_code != 200:
			print(f'Failed to get play by play. Game: {g}, Time: {dt}')
			continue

		p_json = playbyplay_resp.json()

		plays = p_json["game"]["actions"]

		playbyplay_rows = []

		for p in plays:
			action = []
			for a in action_attributes:
				if a in p:
					action.append(p[a])
				else:
					action.append(np.nan)
			playbyplay_rows.append(action)

		playbyplay_df = pd.DataFrame(playbyplay_rows, columns=action_attributes)
		pickle.dump(playbyplay_df, open(PLAYBYPLAY_PREF + "/" + DATE_STR + "/" + str(g) + "/" + time_human + ".pickle", "wb"))
	
	time.sleep(LOOP_FREQ - ((time.time() - START_TIME) % LOOP_FREQ))
	