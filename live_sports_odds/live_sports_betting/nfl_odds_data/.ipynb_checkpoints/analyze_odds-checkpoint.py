import pandas as pd
import pickle
import datetime
import time
import json
import glob


odds_df = pickle.load(open("odds_df_late_games_10_17.pickle", "rb"))

print(odds_df)