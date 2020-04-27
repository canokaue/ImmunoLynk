# ImmunoLynk: COVID19 Immunity Testing Results on the Blockchain
MIT COVID Hack Challenge 2020: Team #thedreamteam

This is a simple but effective implementation of a blockchain-based descentralized test results validator.

The project is divided into the sections bellow.

<img src="designs/cover.jpg" align="center" />

## Immuno Lynk backend (server)

Server built on Flask using an HTTP API. All requests are handled then sent to an IPFS Blockchain instance running on AWS EC2. Infura's [API](https://infura.io/docs) was used to connect to a public IPFS node and register the data permanently or temporarily (depends on the PIN variable in server.py).

For production, a private node implementation example is providade in the /private_ipfs_node folder, courtesy of [dfile.app](dfile.app).

The endpoins are:

#### Image upload

Endpoint:
```bash
http://ec2-3-15-190-197.us-east-2.compute.amazonaws.com/image
```

Response:
```bash
{
    "Hash": "Qme9vV3FULEMiggF3i3fecvD8JQ5ysiAoZhyuTQDFkViWR",
    "Size": "9940",
    "id": "milos",
    "link": "https://ipfs.infura.io:5001/api/v0/cat?arg=Qme9vV3FULEMiggF3i3fecvD8JQ5ysiAoZhyuTQDFkViWR",
    "timestamp": "2020-04-05 13:16:16.393551",
    "type": "IMG"
}
```

#### Data upload

Endpoint:
```bash
http://ec2-3-15-190-197.us-east-2.compute.amazonaws.com/json
```

Response:
```bash
{
    "Hash": "Qme9vV3FULEMiggF3i3fecvD8JQ5ysiAoZhyuTQDFkViWR",
    "Size": "9940",
    "id": "milos",
    "is_immune": true,
    "link": "https://ipfs.infura.io:5001/api/v0/cat?arg=Qme9vV3FULEMiggF3i3fecvD8JQ5ysiAoZhyuTQDFkViWR",
    "scan_hash": "6c8whrnwr8w9eb6wb8erw",
    "timestamp": "2020-04-05 13:16:16.393551",
    "type": "IMG",
    "user_name": "Lon"
}
```

## Immuno Lynk frontend (app)

App built using React components through [expo.io](expo.io) that reads QR Code, automatically takes a snapshot of the test and send's to backend server.

We also have an analytical [dashboard webapp](https://github.com/lon-io/immunolynk-dashboard).

Code is available at [Excellence Ilesamni's repo](https://github.com/lon-io/immunolynk-app).


## Result recognition network (Deep Learning Model)

Model built using Keras and OpenCV2 to detect stripes on the image and determine the test results automatically from the uploaded and processes image.

Full code found in [Veeresh Shringar's repo](https://github.com/VeereshShringari/COVID-testing).

A very simple legacy implementation done at the MIT Hackathon is located at result_recognition_old/, though it's not used anymore.

## Blockchain database alternative

Since images are uploaded in a two-part process (image uploaded as multiform then image data uploadaded as raw json), a more cost effective alternative was also tested using BigchainDB to store the metadata.
