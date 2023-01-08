import pandas as pd
import pickle
import datetime
import time
import json
import glob

# CAN RUN THIS SCRIPT TO AGGREGATE ALL OF THE COLLECTED ODDS RESPONSES (FROM GIVEN DAY) INTO A DATAFRAME

ts = time.time()
time_human = datetime.datetime.fromtimestamp(ts).strftime('%H_%M_%S')
dt = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

book_odds = []

today = date.today()

DATE_STR = today.strftime("%Y%m%d")
ODDS_PREF = "/mnt/storage/data/live_sports/nba/odds/" + DATE_STR

my_respose_files = glob.glob(ODDS_PREF + "/responses/*")

for f in my_respose_files:
	odds_response = pickle.load(open(f, "rb"))
	json_response = json.loads(odds_response.text)
	file_split = f.split("_")
	query_time = int(float(file_split[6][:-7]) * int(1e9))
	for g in json_response:
		game_id = g['id']
		commence_time = g["commence_time"]
		home_team = g["home_team"]
		away_team = g["away_team"]
		bookmakers = g["bookmakers"]
		for book in bookmakers:
			book_name = book["key"]
			book_last_update = book["last_update"]
			markets = book["markets"]
			for m in markets:
				if m["key"] == "h2h":
					for d in m["outcomes"]:
						if d["name"] == home_team:
							home_odds = d["price"]
						if d["name"] == away_team:
							away_odds = d["price"]
			row = [query_time, game_id, commence_time, home_team, away_team, book_name, book_last_update, home_odds, away_odds]
			book_odds.append(row)


odds_df = pd.DataFrame(book_odds, columns = ["query_time", "game_id", "commence_time", "home_team", "away_team", "book_name", "book_last_update", "home_odds", "away_odds"])

pickle.dump(odds_df, open(ODDS_PREF + "/" + time_human + ".pickle", "wb"))