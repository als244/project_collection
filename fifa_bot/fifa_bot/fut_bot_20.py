import fut
import datetime

session = fut.Core(email='andrew@sheinberg.org', passwd='Frenchy99!', secret_answer='codega', cookies=None, platform='xbox', debug=True)

session.keepalive()

now = datetime.datetime.now()
bought = False
while now.minute < 35:
	searchRet = session.search(ctype="player", level="gold", assetId=153079, max_buy=36000)
	for c in searchRet:
		print(c)
		
	# for c in searchRet:
	# 	tradeId = c["tradeId"]
	# 	if session.bid(tradeId, c["buyNowPrice"]):
	# 		print("bought fitness")
	# 		bought = True
	# 		break
	# 	else:
	# 		print("failedBid")
	# 	print()

	# if bought:
	# 	break
	
	now = datetime.datetime.now()
	print("searching for aguero again")


session.logout()