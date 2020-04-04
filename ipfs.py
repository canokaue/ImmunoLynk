import requests
import json
import datetime as dt
import uuid
import sys
import urllib.request

BASE_URL = "https://ipfs.infura.io:5001/api/v0/add?pin=false"
READ_URL = "https://ipfs.infura.io:5001/api/v0/cat?arg="

DOWNLOADS_PATH = 'downloads/'

def upload_ipfs(file_path, file_type, id):
    '''Send POST request to Infura's IPFS API
    '''
    # -F curl flag
    upfile = {'file': (file_path, open(file_path, 'rb'))} 

    # Send and save response
    r = requests.post(BASE_URL, files=upfile)
    response = json.loads(r.text)

    # Add metadata
    meta = dict(type = file_type, id = id, timestamp = str(now))
    response.update(meta)

    # Open past data
    with open('uploads.json', 'r') as uploads:
        upload_data = json.load(uploads)

    # Ugly handling
    new_data = '[' + str(upload_data)[1:-1] + ',' + str(response) + ']'
    new_data = new_data.replace('\'', '"')
    new_dict = json.loads(new_data)

    # Dump new handled data
    with open('uploads.json', 'w') as uploads:
        json.dump(new_dict, uploads)

    return response

def upload_chain(name):
    '''Upload image/json pair
    '''
    upload_ipfs('image.png', 'PNG', name)
    upload_ipfs('data.json', 'JSON', name)
    return

def read_chain(name):
    '''Return chain pair given unique id
    '''
    with open('uploads.json', 'r') as uploads:
        upload_data = json.load(uploads)
    return [upload for upload in upload_data if upload['id'] == name]
    
def download_link(name, extension, download = False):
    '''Build download link for IPFS
    If download = True, also save file to downloads path 
    '''
    with open('uploads.json', 'r') as uploads:
        upload_data = json.load(uploads)
    target = [upload['Hash'] for upload in upload_data if upload['id'] == name and upload['type'] == extension]
    url = READ_URL + str(target[0])
    if download:
        urllib.request.urlretrieve(url,DOWNLOADS_PATH + name + '.' + extension.lower())
    return url

if __name__ == '__main__':
    now = dt.datetime.now()
    uhash = uuid.uuid4().hex # Unique hash for each pair
    upload_chain(uhash)
    image, data = read_chain(uhash)
    print(download_link(uhash, 'JSON'))


