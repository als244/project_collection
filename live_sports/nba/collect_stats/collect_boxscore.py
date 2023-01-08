import requests
import json
import pickle
import datetime
import time
import pandas as pd
import os
from datetime import date

today = date.today()

SCOREBOARD_PREF = "/mnt/storage/data/live_sports/nba/stats/scoreboards"
DATE_STR = today.strftime("%Y%m%d")

scoreboard_df = pickle.load(open(SCOREBOARD_PREF + "/" + DATE_STR + ".pickle", "rb"))

game_ids = scoreboard_df["game_id"]

BOXSCORE_PREF = "/mnt/storage/data/live_sports/nba/stats/boxscores/"

if not os.path.exists(BOXSCORE_PREF + DATE_STR):
	os.makedirs(BOXSCORE_PREF + DATE_STR)

LOOP_FREQ = 30.0

START_TIME = time.time()

### KEEP SCRIPT RUNNING DURING GAMES TO COLLECT UPDATED BOXSCORES

while True:
	ts = time.time()
	time_human = datetime.datetime.fromtimestamp(ts).strftime('%H_%M_%S')
	dt = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

	boxscore_rows = []

	stats = ["assists", "benchPoints", "blocks", "fieldGoalsAttempted", "fieldGoalsMade", "foulsOffensive", "foulsDrawn", "foulsPersonal", "foulsTeam", "freeThrowsAttempted", "freeThrowsMade", 
				"pointsFastBreak", "pointsFromTurnovers", "pointsInThePaint", "reboundsDefensive", "reboundsOffensive", "reboundsTotal", "secondChancePointsAttempted", "secondChancePointsMade", "steals", 
				"threePointersAttempted", "threePointersMade", "turnovers", "twoPointersAttempted", "twoPointersMade"]
	
	for g in game_ids:

		req_str = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_" + str(g) + ".json"
		boxscore_resp = requests.get(req_str)

		# happens for games that haven't started
		if boxscore_resp.status_code != 200:
			print(f'Failed to get boxscore. Game: {g}, Time: {dt}')
			continue

		b_json = boxscore_resp.json()

		b = b_json["game"]

		game_boxscore = [b["gameId"], b["gameStatusText"], b["gameClock"], b["homeTeam"]["teamId"], b["homeTeam"]["teamTricode"], b["homeTeam"]["score"], b["awayTeam"]["teamId"], b["awayTeam"]["teamTricode"], b["awayTeam"]["score"]]
		for t in ["homeTeam", "awayTeam"]:
			for s in stats:
				game_boxscore.append(b[t]["statistics"][s])

		
		boxscore_rows.append(game_boxscore)

	my_columns = ["gameId", "gameStatusText", "gameClock", "home_teamId", "home_teamTricode", "home_score", "away_teamId", "away_teamTricode", "away_score"]
	for t in ["home", "away"]:
		for s in stats:
			my_columns.append(t + "_" + s)

	boxscore_df = pd.DataFrame(boxscore_rows, columns=my_columns)
	pickle.dump(boxscore_df, open(BOXSCORE_PREF + DATE_STR + "/" + time_human + ".pickle", "wb"))

	time.sleep(LOOP_FREQ - ((time.time() - START_TIME) % LOOP_FREQ))