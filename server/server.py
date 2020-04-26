import ipfs
from flask import Flask, jsonify,request
import os
import json
from tempfile import NamedTemporaryFile
from result_recognition import TestResModel
from keras_applications.resnet import ResNet50


server = Flask(__name__)

PORT = '5000'

THREAD_RUN = False # keep false for model

DEBUG_RUN = False

HOST = '0.0.0.0'

MYIP = ''

_current_imagepath = ''
_current_datapath = ''

model = TestResModel(engine=ResNet50, input_dims=(224, 224, 3), batch_size=3,
                     learning_rate=5e-4, n_augment = 0,
                    num_epochs=12, decay_rate=0.8, decay_steps=1, weights="imagenet", verbose=2)

model.load('resnet50_weights.h5')


@server.route('/image', methods=['POST'])
def receive_image():
    try: 
        # Get request file and save to server
        image = request.files['file']
        filename = image.filename
        _current_imagepath = ipfs.DOWNLOADS_PATH + str(filename)
        image.save(_current_imagepath)

        # Upload to blockchain, save current data and return link to client
        response = ipfs.upload_ipfs(_current_imagepath, 'IMG', _current_imagepath)
        message = clean_response(response)

        # Predict image test result and append to message
        result = model.predict(filename)
        message['result'] = str(result)

        with open ('current_data.json', 'w') as cd:
            json.dump(message, cd)
        return jsonify(message)
    except Exception as e:
        message = 'Error receiving message: % s' % e
        with open ('error.json', 'w') as err:
            json.dump(message, err)
        return jsonify({'status' : message})


def clean_response(response):
    response.pop('Name')
    response['id'] = response['id'].replace('downloads/', '').split('.')[0]
    link = dict(link = ipfs.READ_URL + response['Hash'])
    response.update(link)
    return response

@server.route('/json', methods=['POST'])
def process_json():
    input_json = request.get_json()
    with open ('current_data.json', 'r') as cd:
        img_data = json.load(cd)
    input_json.update(img_data)
    return jsonify(input_json)


if __name__ == '__main__':
    server.run(host=HOST, debug=DEBUG_RUN, port=PORT, threaded=THREAD_RUN)


# .json input example input - use for debugging
'''
{
   "scan_hash" : "6c8whrnwr8w9eb6wb8erw",
   "user_name": "Lon",
   "is_immune": true,
   "timestamp": 1586042898041
}
'''