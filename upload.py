import requests
import json

BASE_URL = "https://ipfs.infura.io:5001/api/v0/add?pin=false"

with open ('data.json', 'r') as read_data:
    json_data = json.load(read_data)