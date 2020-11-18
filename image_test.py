from __future__ import print_function
import requests
import json
import os

addr = 'http://127.0.0.1:5005'
test_url = addr +'/find-operators'


## testing
image_path = './images/local-filename.jpg'
with open(image_path, 'rb') as img:
    name_img = os.path.basename(image_path)
    files = {'image': (name_img, img, 'multipart/form-data', {'Expires':'0'})}
    r = requests.post(test_url, files=files)
    print(r.status_code)
    print(json.loads(r.text))