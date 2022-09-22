import pandas as pd
import pickle
import datetime
import time
import json
import glob


book_odds = []

my_respose_files = glob.glob("./odds_resp_*")

for f in my_respose_files:
	odds_response = pickle.load(open(f, "rb"))
	json_response = json.loads(odds_response.text)
	file_split = f.split("_")
	query_time = int(float(file_split[4][:-7]) * int(1e9))
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

pickle.dump(odds_df, open("odds_df_ncaab_games_03_17.pickle", "wb"))