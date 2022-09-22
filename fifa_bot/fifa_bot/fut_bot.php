<?php
require __DIR__.'/vendor/autoload.php';
use Shapecode\FUT\Client\Api\Core;
use Shapecode\FUT\Client\Exception\FutException;
use Shapecode\FUT\Client\Authentication\Account;
use Shapecode\FUT\Client\Authentication\Credentials;
use Shapecode\FUT\Client\Authentication\Session;
use Shapecode\FUT\Client\Response;
use Shapecode\FUT\Client\Items;
$credentials = new Credentials("andrew@sheinberg.org", "Frenchy99!", "xbox");


$session = null;
$account = new Account($credentials, $session);
$fut = new Core($account);


try {
	$login = $fut->login($code);
	$session = $account->getSession();
} catch(FutException $e) {
	$reason = $e->getReason();
	die("We have an error logging in: " . $reason . "\n");
}

// parameters for each
// assetId = 'maskedDefId'
// min/max buy = 'minb' / 'maxb'
// min/max bid = 'micr' / 'macr'
// nationality = 'nat'
// club = 'team'
// league = 'leag'
// position = 'pos'
// for rare, set 'rare' = 'SP'
// level = 'lev'
// category = 'cat'


//$searchOptions = ["type" => "development", "maskedDefId" => 5002006, "maxb" => 850, "lev" => "gold", "cat" => "fitness"];  // gold squad fitness

// duration in seconds
//iterateSearch($searchOptions, 8000, true, false, true, $fut);
//mySearch($searchOptions, true, false, $fut);

//testWatchlist($sokratisId, $cheapestEstimate, $fut);
//iterateBids($sokratisId, $cheapestEstimate, $margin, 240, $fut);


//$terStegenId = 192448;


$kanteId = 215914;
$duration = 10800;

//$cheapest = findCheapestPlayerBuyNow($terStegenId, 47000, $fut);
//print("Cheapest Ter Stegen is: " . $cheapest . "\n");
findAtLevel($kanteId, 130000, 15000000, time(), $duration, 1, $fut);


// UTILITY FUNCTIONS


function logout(Core $fut){
	$fut->logout();
}

function testWatchlist(int $itemId, int $startingPoint, Core $fut){
	$watchlist = $fut -> watchlist() -> getAuctions();
	$margin = 600;
	$cheapestBin = findCheapestPlayerBuyNow($itemId, $startingPoint, $fut);
	print("Chepest Bin" . $cheapestBin . "\n");
	foreach ($watchlist as $listing) {
		if ($listing -> getTradeState() == 'closed' || $listing -> getBidValue() > $cheapestBin - $margin){
			print("Removing from Watchlist");
			$fut -> watchlistDelete($listing->getTradeId());
		}
	}
}


function iterateBids(int $itemId, $cheapestEstimate, int $margin, int $duration, Core $fut){
	$time = time();
	$endTime = $time + $duration;
	while ($time < $endTime){
		try{ 

			// clear watchlist
			if ($time > $prevWatchTime + 60){
				$watchlist = $fut -> watchlist() -> getAuctions();
				foreach ($watchlist as $listing) {
					if ($listing -> getTradeState() == 'closed' || $listing -> getBidValue() > $cheapestBin - $margin){
						$fut -> watchlistDelete($listing->getTradeId());
					}
				}
				$prevWatchTime = $time;
			}
			// clear tradepile
			$tradepile = $fut->tradepile()->getAuctions();
			$nSold = 0;
			$myAuctions = [];
			$nSold = 0;
			$nListed = 0;
			foreach ($tradepile as $auction) {
				if ($auction->getTradeState() == 'closed'){
					$fut->removeSold($auction->getTradeId());
					$nSold += 1;
				}
				elseif ($auction->getTradeState() == 'active'){
					$nListed += 1;
				}
			}
			if ($nSold > 0){
				print("Just sold: " . $nSold . "\n");
			}
			if (count($myAuctions) > 0){
				print("Still are selling: " . count($myAuctions) . "\n");
			}

			// only let there be 30 unlisted items in tradepile
			if (count($tradepile) - $nSold - $nListed < 30){
				$nPlaced = placeBids($itemId, $cheapestEstimate, $margin, $fut);
				print("Placed " . $nPlaced . " bids\n");
			}
			//only want to sell 5 listings at a time
			if ($sell && $nListed < 5){
				$newListed = 0;
			 	foreach ($tradepile as $auction) {
			 		if ($nListed + $newListed < 5 && $auction->getTradeState() != 'closed' && $auction->getTradeState() != 'active'){
			 			$auctionId = $auction->getTradeId();
			 			$card = $auction->getItem();
			 			$cardId = $card->getItemId();
			 			$cheap = $findCheapestPlayerBuyNow($cardId, $cheapestEstimate, $fut);
			 			mySell($cardId, $cheap - 100, $cheap, $fut);
			 			$newListed += 1;
			 		}
			 		elseif ($nListed + $newListed >= 5) {
			 			break;
			 		}
			 	}
			}
		} catch (SessionExpiredException $e){
			print("Session Expired");
			$fut -> resetSession();
		}
		$time = time();
	}

}


function findCheapestPlayerBuyNow(int $itemId, int $upperBound, Core $fut){
	$searchOptions = ["maskedDefId" => $itemId, "maxb" => $upperBound];
	$items = $fut->search($searchOptions);
	$listings = $items -> getAuctions();
	$min = $upperBound + 1;
	$max = 0;
	foreach ($listings as $l){
		if ($l->getBuyNowPrice() < $min) {
			$min = $l->getBuyNowPrice();
		}
		if ($l->getBuyNowPrice() > $max){
			$max = $l->getBuyNowPrice();
		}
	}
	if ($min == $upperBound + 1){
		return findCheapestPlayerBuyNow($itemId, $upperBound + 500, $fut);
	}
	if (count($listings) == 20 && $max == $min){
		return findCheapestPlayerBuyNow($itemId, $upperBound - 500, $fut);
	}
	return $min;

}

function findAtLevel(int $itemId, int $upperBound, int $maxBid, int $startTime, int $duration, int $cnt, Core $fut){
	date_default_timezone_set("America/New_York");
	if (time() > $startTime + $duration){
		return;
	}
	if ($cnt % 60 == 0){
		print("Searched " . $cnt . " times. It is now " . date("h:i:sa") . "\n");
		sleep(15);
	}

	$searchOptions = ["maskedDefId" => $itemId, "macr" => $maxBid, "maxb" => $upperBound];
	$items = $fut->search($searchOptions);
	$listings = $items -> getAuctions();
	if (count($listings) == 0){
		print("NONE found\n");
	} else{
		print(count($listings) . " found at " . date("h:i:sa") . ":\n");
	}
	foreach ($listings as $l){
		print($l->getBuyNowPrice() . " available\n");
	}
	findAtLevel($itemId, $upperBound, $maxBid - 1000, $startTime, $duration, $cnt + 1, $fut);
}

function placeBids(int $itemId, int $startingPoint, int $margin, Core $fut){

	$cheapestBin = findCheapestPlayerBuyNow($itemId, $startingPoint, $fut);
	print("Cheapest Buy Now " . $cheapestBin . "\n");
	$bidSearchOptions = ["maskedDefId" => $itemId, "macr" => $cheapestBin - $margin];
	$searchResults = $fut->search($bidSearchOptions);
	$listings = $searchResults->getAuctions();
	$nPlaced = 0;
	foreach ($listings as $listing) {
		// only bid when less than 2 minutes
		if ($listing->getExpires() < 120){
			$curBid = $listing->getBidValue();
			$auctionId = $listing->getTradeId();
			$card = $listing->getItem();
			$cardId = $card->getItemId();
			$ret = myBid($listing->getTradeId(), $cardId, $curBid + 500, $fut);
			if ($ret){
				$nPlaced += 1;
			}
		}
	}
	return $nPlaced;
}


function myBid(int $auctionId, int $itemId, int $purchPrice, Core $fut){
	$credits = $fut->credits();
	if ($credits > $purchPrice){
		print("Trying to bid for " . $purchPrice . "\n");
		$res = $fut->bid($auctionId, $purchPrice);
		if ($res->getCredits() == -1){
			print("Failed Bid" . "\n");
			return false;
		} else{
			print("Successful Bid " . $purchPrice . "\n");
		}
	}
	return true;
}

function mySell(int $itemId, int $bid, int $bin, Core $fut, int $duration = 3600){
	$fut->sell($itemId, $bid, $bin);
}

function snipe(int $itemId, int $maxbin, int $duration, Core $fut){

}



// function iterateSearchOld(array $searchOptions, int $duration, bool $print, bool $buy, bool $sell, Core $fut){
// 	$time = time();
// 	$prevClear = 0;
// 	$endTime = $time + $duration;
// 	while ($time < $endTime){
// 		try{ 
// 			// clear tradepile every 20 seconds
// 			$tradepile = $fut->tradepile()->getAuctions();
// 			if($time  > $prevClear){
// 				$nSold = 0;
// 				$myAuctions = [];
// 				$inactive = 0;
// 				foreach ($tradepile as $auction) {

// 					if ($auction->getTradeState() == 'closed'){
// 						$fut->removeSold($auction->getTradeId());
// 						$nSold += 1;
// 					}
// 					elseif ($auction->getTradeState() == 'active'){
// 						$sellPrice = $auction->getBuyNowPrice();
// 						if (array_key_exists($sellPrice, $myAuctions)){
// 							$myAuctions[$sellPrice] += 1;
// 						}
// 						else{
// 							$myAuctions[$sellPrice] = 1;
// 						}
// 					}
// 					else{
// 						$inactive += 1;
// 					}
// 				}
// 				if ($nSold > 0){
// 					print("Just sold: " . $nSold . "\n");
// 				}
// 				if (count($myAuctions) > 0){
// 					print("Still are selling: " . count($myAuctions) . "\n");
// 					// foreach ($myAuctions as $key => $value) {
// 					// 	print($key . ": " . $value . "\n");
// 					// }
// 				}
// 				// if (count($tradepile) - $nSold - array_sum($myAuctions) > 0){
// 				// 	print(count($tradepile) - $nSold - array_sum($myAuctions) . " unilisted card in tradepile\n");
// 				// }
				
// 				$prevClear = $time;
// 			}

// 			// only let there be 10 unlisted items in tradepile
// 			if (count($tradepile) - $nSold - array_sum($myAuctions) < 30){
// 				$boughtAt = mySearch($searchOptions, $print, $buy, $myAuctions, $fut);
// 				if (count($boughtAt) > 0){
// 					print("Bought:\n");
// 					foreach ($boughtAt as $key => $value) {
// 						print($key . ": ".  $value . "\n");
// 					}
// 				}
// 			}
// 			//only want to sell 10 listings at a time
// 			if ($sell && array_sum($myAuctions) < 5){
// 				$nListed = 0;
// 			 	foreach ($tradepile as $auction) {
// 			 		if ($nListed + array_sum($myAuctions) < 5 && $auction->getTradeState() != 'closed' && $auction->getTradeState() != 'active'){
// 			 			$auctionId = $auction->getTradeId();
// 			 			$card = $auction->getItem();
// 			 			$cardId = $card->getItemId();
// 			 			// $starts = range(150, 700, 50);

// 			 			// $startInd = array_rand($starts, 1);
// 			 			// print($startInd[0]);
// 			 			// $bidStart = $starts[$startInd];
// 			 			mySell($cardId, 750, 850, $fut);
// 			 			$nListed += 1;
// 			 		}
// 			 		elseif ($nListed + array_sum($myAuctions) >= 5) {
// 			 			break;
// 			 		}
// 			 	}
// 			 }
// 		} catch (SessionExpiredException $e){
// 			print("Session Expired");
// 			$fut -> resetSession();
// 		}
// 		$time = time();
// 	}
// }



// function mySearchOld(array $searchOptions, bool $print, bool $buy, array $myAuctions, Core $fut){
// 	print("Searching...\n");
// 	$items = $fut->search($searchOptions);
// 	$listings = $items -> getAuctions();
// 	$cnt =count($listings);
// 	print("Found: " . $cnt . "\n");
	
// 	$boughtAt = [];

// 	usort($listings, function($a, $b) {
// 		if ($a->getBuyNowPrice() == $b->getBuyNowPrice()){
// 			if ($a->getExpires() ==  $b->getExpires()) {
// 				return 0;
// 			}
//       		return ($a->conditional < $b->conditional) ? 1 : -1;
// 		}
// 	 	return $a->getBuyNowPrice() - $b->getBuyNowPrice();
// 	});

// 	// set price level too high
// 	if ($cnt == 20 && $listings[0]->getBuyNowPrice() == end($listings)->getBuyNowPrice()){
// 		$newOptions = $searchOption;
// 		$newOptions['maxb'] -= 50;
// 		return mySearch($newOptions, $print, $buy, $myAuctions, $fut);
// 	}

// 	if ($print){
// 		$toPrint = $listings;
// 	 	foreach ($listings as &$listing){
// 	 		print($listing->getBuyNowPrice());
// 	 		print("\n");
// 	 		print("Expires: " . $listing->getExpires() . "\n");
// 		}
// 	}
// 	if ($buy){
// 		foreach ($listings as $listing) {
// 			print("Expires: " . $listing->getExpires() . "\n");
// 			if ($listing->getExpires() < 3595){
// 				continue;
// 			}
// 			$binPrice = $listing->getBuyNowPrice();
// 			$auctionId = $listing->getTradeId();
// 			$card = $listing->getItem();
// 			$cardId = $card->getItemId();
// 			$ret = myBuyNow($auctionId, $cardId, $binPrice, $fut);
// 			if ($ret){
// 				if (array_key_exists($binPrice, $boughtAt)){
// 					$boughtAt[$binPrice . ''] += 1;
// 				}
// 				else{
// 					$boughtAt[$binPrice . ''] = 1;
// 				}
// 			}
// 		}
// 	}
// 	return $boughtAt;
// }
	
	// // set max buy too low
	// if ($cnt == 0){
	// 	$searchOptions['maxb'] += 50;
	// 	return mySearch($searchOptions, $print, $buy, $myAuctions, $fut);
		
	// }

	// usort($listings, function($a, $b) {
	// 	return $a->getBuyNowPrice() - $b->getBuyNowPrice();
	// });

	// // set max buy too high
	// if ($cnt == 20 && end($listings)->getBuyNowPrice() - $listings[0]->getBuyNowPrice() <= 100){
	// 	$searchOptions['maxb'] -= 50;
	// 	return mySearch($searchOptions, $print, $buy, $myAuctions, $fut);
	// }

	// if ($print){
	// 	foreach ($listings as &$listing){
	// 		print($listing->getBuyNowPrice());
	// 		print("\n");
	// 	}
	// }

	// // if (!$buy){
	// // 	return;
	// // }

	// // BUYING CARDS
	
	// // Tracking what would have been bought
	// $boughtAt = [];
	
	// // Gettting range of prices on market
	// $prices = [];
	// $prevPrice = 0;
	// foreach ($listings as $listing) {
	// 	$binPrice = $listing -> getBuyNowPrice();
	// 	array_push($prices, $binPrice);
	// 	$prevPrice = $binPrice;
	// }
	// $priceCounts = array_count_values($prices);
	// $prices = array_unique($prices);


	// See we need to monopolize

	// NEED TO THINK MORE ABOUT LOGIC OF BUYING MULTIPLE CARDS AT SAME TIME

	// $increase = 100;
	// $nextBetterOptions = $searchOptions;
	// $nextBetterOptions['maxb'] = $prices[0] + $increase;
	// $nextBetter = $fut->search($nextBetterOptions);
	// $nextCnt = count($nextBetter -> getAuctions());
	// print("The count at level of " . $nextBetterOptions['maxb'] . " is: " . $nextCnt . "\n");
	// if ($nextCnt - $priceCounts[$prices[0]] <= 3){
	// 	for ($i = 1; $i < count($prices); $i++) {
	// 		if ($priceCounts[$prices[$i]] > 1){
	// 			$cutoffPrice = $prices[$i];
	// 			break;
	// 		}
	// 	}
	// 	foreach ($listings as $listing) {
	// 		$binPrice = $listing->getBuyNowPrice();
	// 		if ($cutoffPrice - $binPrice >= 100){
	// 			$auctionId = $listing->getTradeId();
	// 			$card = $listing->getItem();
	// 			$cardId = $card->getItemId();
	// 			// only buying 4 total
	// 			if	(array_sum($boughtAt) <= 3){
	// 				$boughtAt[$binPrice] += 1;
	// 				//myBuyNow($auctionId, $cardId, $binPrice, $fut);
	// 			}
	// 		}
	// 	}
	// } 


	// FINDING UNDERPRICED cards

	// // don't bother trying to buy if differce betewen lowest and highest is only 100
	// if (max($prices) - min($prices) <= 100){
	// 	return [];
	// }
	
	// // BUYING CONDITION

	// // if there is only 1, buy it and sell it as m
	// if (count($listings) == 1){
	// 	$binPrice = $listings[0] -> getBuyNowPrice();

	// 	// simple case of selling for min(bin + 100, maxb - bin greater than purchase price
	// 	if ($searchOptions['maxb'] - $binPrice >= 50){
	// 		$binPrice = $listing->getBuyNowPrice();
	// 		$auctionId = $listing->getTradeId();
	// 		$card = $listing->getItem();
	// 		$cardId = $card->getItemId();
	//		$ret = myBuyNow($auctionId, $cardId, $binPrice, $fut);
	// 		if ($ret){
	// 			$boughtAt[$binPrice] += 1;
	// 		}

	// 	}
	// 	return $boughtAt;

	// }

	// // Cutoff is determined when after first occurance of difference between i + 1 and i elements being smaller than 150
	// // maybe more logic to buy in chunks? find large gap between low and high prices
	// // BUY ITEMS < $cutoffPrice
	// $ind = 0;
	// while ($ind < count($prices) - 1){
	// 	if ($prices[$ind + 1] - $prices[$ind] >= 150){
	// 		$ind += 1;
	// 	}
	// 	else{
	// 		break;
	// 	}
	// }


	// $cutoffPrice = $prices[$ind];

	// // also don't undercut myself
	// if (count($myAuctions) > 0 && min($myAuctions) < $cutoffPrice){
	// 	$cutoffPrice = min($myAuctions);
	// }

	// print("Cutoff Price: " . $cutoffPrice . "\n");

	// foreach ($listings as $listing) {
	// 	$binPrice = $listing->getBuyNowPrice();
	// 	$auctionId = $listing->getTradeId();
	// 	$card = $listing->getItem();
	// 	$cardId = $card->getItemId();

	// 	// using buying condition and also limiting to only buy 4 items
	// 	if ($binPrice < $cutoffPrice && array_sum($boughtAt) <= 3){
	// 		$ret = myBuyNow($auctionId, $cardId, $binPrice, $cutoffPrice - 50, $fut);
	// 		if ($ret){
	// 			$boughtAt[$binPrice] += 1;
	// 		}
	// 	}
	// }
	//return $boughtAt;
//}






// $options = ["maskedDefId" => 153079, "maxb" => 50000];
// $items = $fut->search($options);
// $listings = $items -> getAuctions();
// usort($listings, function($a, $b) {
//     return $a->getBuyNowPrice() - $b->getBuyNowPrice();
// });
// foreach ($listings as &$listing){
// 	print($listing->getBuyNowPrice());
// 	print("\n");
// }
?>