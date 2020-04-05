# DFile: A fancy IPFS-based file sharing mode

This is an IPFS-based file hosting that also runs on [https://dfile.app](https://dfile.app)

## DFile backend (server)

Before running the service for the first time, run

```bash
cp config.sample config.py
```

Modify config.py (mainly setup your [IPFS Node](https://docs.ipfs.io/introduction/usage/))

```bash
IPFS_CONNECT_URL = "/ip4/127.0.0.1/tcp/5001/http"
IPFS_FILE_URL = "http://127.0.0.1:8080/ipfs/"
DOMAIN = "http://localhost:5000"
```

Run it

```bash
./dfile.py debug
```

## How to use

```bash
# Upload using cURL
➜ curl -F file=@yourfile.txt https://dfile.app
https://dfile.app/QmV...HZ
# Download the file
➜ curl https://dfile.app/QmV...HZ -o yourfile.txt
```