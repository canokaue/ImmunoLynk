from bigchaindb_driver import BigchainDB
from bigchaindb_driver.crypto import generate_keypair

bdb = BigchainDB('https://test.bigchaindb.com')
client = generate_keypair()
data = { 'entry' : 2983162837162,
	'id' : 'ya6dgia6hdei66e7382k',
	'geo' : 74928364283464,
	'scan_hash' : '6c8whrnwr8w9eb6wb8erw',
	'age' : 23,
	'gender' : 'm',
	'symptom_day' : 14,
	'symptoms' : 10001010,
	'income_range' : 4
}
tx = bdb.transactions.prepare(
    operation='CREATE',
    signers=client.public_key,
    asset={'data': data})
signed_tx = bdb.transactions.fulfill(
    tx,
    private_keys=client.private_key)
bdb.transactions.send_commit(signed_tx)