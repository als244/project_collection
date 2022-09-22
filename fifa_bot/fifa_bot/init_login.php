<?php
require __DIR__.'/vendor/autoload.php';
use Shapecode\FUT\Client\Api\Core;
use Shapecode\FUT\Client\Exception\FutException;
use Shapecode\FUT\Client\Authentication\Account;
use Shapecode\FUT\Client\Authentication\Credentials;
use Shapecode\FUT\Client\Authentication\Session;
$credentials = new Credentials("andrew@sheinberg.org", "Frenchy99!", "xbox");


$session = null;
$account = new Account($credentials, $session);
$fut = new Core($account);


try {
    $login = $fut->login($code);
    $session = $account->getSession();
} catch(FutException $e) {
    $reason = $e->getReason();
    die("We have an error logging in: ".$reason);
}

$options = ["assetId" => 153079];
$items = $fut->search($options);
$listings = $items -> getAuctions();
foreach ($listings as &$listing){
	print($listing->getBuyNowPrice());
	print("\n");
	$item = $listing->getItem();
	print($item->getRating());
	print("\n");
	print($item->getNation());
	print("\n");
	print("\n");
}
?>