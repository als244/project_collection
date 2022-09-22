<?php
require __DIR__.'/vendor/autoload.php';
use Shapecode\FUT\Client\Api\Core;
use Shapecode\FUT\Client\Exception\FutException;
use Shapecode\FUT\Client\Authentication\Account;
use Shapecode\FUT\Client\Authentication\Credentials;
use Shapecode\FUT\Client\Authentication\Session;
include 'init_login.php';
include '/vendor/shapecode/fut-api/src/Api/AbstractCore.php';
$code = $argv[1];
echo $code;
completeLogin($code);
function completeLogin(string $code){
	try {
		$login = useCodeForEA($code);
	} catch(FutException $e) {
		$reason = $e->getReason();
		die("We have an error logging in: ".$reason);
	}
}

function useCodeForEA(string $code){
	if (strpos($responseContent, $locale->get('login.security_code')) !== false) {
		if ($code === null) {
			throw new ProvideSecurityCodeException($call->getResponse());
		}

		$headers['Referer'] = (string) $url;
		$call               = $fut->simpleRequest('POST', str_replace('s3', 's4', (string) $url), [
			'headers'     => $headers,
			'form_params' => [
				'oneTimeCode'      => $code,
				'_trustThisDevice' => 'on',
				'trustThisDevice'  => 'on',
				'_eventId'         => 'submit',
			],
			'on_stats'    => static function (TransferStats $stats) use (&$url) : void {
				$url = $stats->getEffectiveUri();
			},
		]);
		$responseContent    = $call->getContent();

		if (strpos($responseContent, $locale->get('login.incorrect_code_1')) !== false) {
			throw new IncorrectSecurityCodeException($call->getResponse());
		}

		if (strpos($responseContent, $locale->get('login.incorrect_code_2')) !== false) {
			throw new IncorrectSecurityCodeException($call->getResponse());
		}
	}
	if ($url->getFragment() === '') {
		throw new AuthFailedException($call->getResponse());
	}
	parse_str($url->getFragment(), $matches);

	$accessToken = $matches['access_token'];
	$tokenType   = $matches['token_type'];
	$expiresAt   = new DateTime('+' . $matches['expires_in'] . ' seconds');

	$fut->simpleRequest('GET', 'https://www.easports.com/fifa/ultimate-team/web-app/');

	$headers['Referer']       = 'https://www.easports.com/fifa/ultimate-team/web-app/';
	$headers['Accept']        = 'application/json';
	$headers['Authorization'] = $tokenType . ' ' . $accessToken;

	$call            = $fut->simpleRequest('GET', 'https://gateway.ea.com/proxy/identity/pids/me', [
		'headers' => $headers,
	]);
	$responseContent = json_decode($call->getContent(), true, 512, JSON_THROW_ON_ERROR);

	$nucleus_id = $responseContent['pid']['externalRefValue'];
	$dob        = $responseContent['pid']['dob'];

	unset($headers['Authorization']);

	$headers['Easw-Session-Data-Nucleus-Id'] = $nucleus_id;

	        //shards
	try {
		$fut->simpleRequest('GET', 'https://' . $fut::AUTH_URL . '/ut/shards/v2', [
			'headers' => $headers,
		]);
	} catch (RequestException $e) {
		throw new ServerDownException($e->getResponse(), $e);
	}

	        //personas
	try {
		$call            = $fut->simpleRequest('GET', $fut->getFifaApiUrl() . '/user/accountinfo', [
			'headers' => $headers,
			'query'   => [
				'filterConsoleLogin'    => 'true',
				'sku'                   => $fut::SKU,
				'returningUserGameYear' => '2019',
			],
		]);
		$responseContent = json_decode($call->getContent(), true);
	} catch (ConnectException $e) {
		throw new ServerDownException($e->getResponse(), $e);
	}

	if (! isset($responseContent['userAccountInfo']['personas'])) {
		throw new NoPersonaException($call->getResponse());
	}

	$personasValues = array_values($responseContent['userAccountInfo']['personas']);
	$persona        = array_pop($personasValues);
	$persona_id     = $persona['personaId'] ?? null;

	        //validate persona found.
	if ($persona_id === null) {
		throw new NoPersonaException($call->getResponse());
	}

	        //validate user state
	if ($persona['userState'] === 'RETURNING_USER_EXPIRED') {
		throw new UserExpiredException($call->getResponse());
	}

	        //authorization
	unset($headers['Easw-Session-Data-Nucleus-Id']);
	$headers['Origin'] = 'http://www.easports.com';

	$call            = $fut->simpleRequest('GET', 'https://accounts.ea.com/connect/auth', [
		'headers' => $headers,
		'query'   => [
			'client_id'     => 'FOS-SERVER',
			'redirect_uri'  => 'nucleus:rest',
			'response_type' => 'code',
			'access_token'  => $accessToken,
			'release_type'  => 'prod',
		],
	]);
	$responseContent = json_decode($call->getContent(), true);

	$auth_code = $responseContent['code'];

	$headers['Content-Type'] = 'application/json';
	$call                    = $fut->simpleRequest('POST', $fut->getFutAuthUrl(), [
		'headers' => $headers,
		'body'    => json_encode([
			'isReadOnly'       => false,
			'sku'              => $fut::SKU,
			'clientVersion'    => $fut->clientVersion,
			'nucleusPersonaId' => $persona_id,
			'gameSku'          => $fut->getGameSku(),
			'locale'           => 'en-US',
			'method'           => 'authcode',
			'priorityLevel'    => 4,
			'identification'   => [
				'authCode'    => $auth_code,
				'redirectUrl' => 'nucleus:rest',
			],
		], JSON_THROW_ON_ERROR),
	]);

	if ($call->getResponse()->getStatusCode() === 401) {
		throw new MaxSessionsException($call->getResponse());
	}

	if ($call->getResponse()->getStatusCode() === 500) {
		throw new ServerDownException($call->getResponse());
	}

	$responseContent = json_decode($call->getContent(), true, 512, JSON_THROW_ON_ERROR);
	if (isset($responseContent['reason'])) {
		switch ($responseContent['reason']) {
			case 'multiple session':
			case 'max sessions':
			throw new MaxSessionsException($call->getResponse());
			case 'doLogin: doLogin failed':
			throw new AuthFailedException($call->getResponse());
			default:
			throw new FutResponseException($responseContent['reason'], $call->getResponse());
		}
	}

	$phishingToken = $responseContent['phishingToken'];
	$sid           = $responseContent['sid'];

	$fut->setSessionData(
		(string) $persona_id,
		$nucleus_id,
		$phishingToken,
		$sid,
		$dob,
		$accessToken,
		$tokenType,
		$expiresAt
	);

	$fut->pin->sendEvent('login', 'success');
	$fut->pin->sendEvent('page_view', 'Hub - Home');

	        // return info
	return [
		'email'          => $credentials->getEmail(),
		'access_token'   => $accessToken,
		'token_type'     => $tokenType,
		'nucleus_id'     => $nucleus_id,
		'persona_id'     => $persona_id,
		'phishing_token' => $phishingToken,
		'session_id'     => $sid,
		'dob'            => $dob,
		'expiresAt'      => $expiresAt,
	];
}
?>