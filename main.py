try:
    from PIL import Image
except ImportError:
    import Image
import cv2
from flask import Flask, request, Response, jsonify
from flask_pymongo import PyMongo
from pymongo.errors import ConnectionFailure, AutoReconnect, ServerSelectionTimeoutError
import urllib.request
from urllib.error import URLError, HTTPError
import imagehash
import json
import os
import numpy as np
import jsonpickle
import time
from modules.match import Match
from stores.image_hash_store import ImageHashStore
from MTM import matchTemplates, drawBoxesOnRGB

MONGO_URL = "127.0.0.1:27017/arcknights"
client = None
store = ImageHashStore()
IMAGE_LOCAL = './images/local-filename.'
TEMPLATE_PATH = './template/'
DEFAULT_IMAGE_HEIGHT = 1080
app = Flask(__name__)


def load_hash():
    # load image dhash to db image size 230 X 370
    images = []
    for r, d, fs in os.walk(TEMPLATE_PATH):
        for img in fs:
            if img != '.DS_Store':
                hashes = {}
                hashes['_id'] = str(imagehash.dhash(Image.open(os.path.join(r, img))))
                # remove jpg
                if '.jpg' in img:
                    hashes['operator'] = img.replace('.jpg', '')
                images.append(hashes)
    return images

# -------------- Test Routes ----------------
@app.route('/', methods=['GET'])
def home():
    return '''<h1>TEST</h1>
<p>The connection works</p>'''



@app.route('/find-operators', methods=['POST'])
def get_image_classification():
    start = time.time()
    try:
        r = request
        nparr = np.frombuffer(r.files['image'].read(), np.uint8)
        # nparr = np.frombuffer(r.data, np.uint8)
        test_img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    except Exception as e:
        raise RuntimeError("Failed to load image data")

    matcher = Match()
    # resize img to match template
    default = matcher.maintain_aspect_ratio_resize(test_img, height=DEFAULT_IMAGE_HEIGHT)
    marker_ratio = matcher.find_ratio(default)
    resize = matcher.maintain_aspect_ratio_resize(default,
                                                  height=int(DEFAULT_IMAGE_HEIGHT / float(marker_ratio)))
    hits = matcher.match_template(resize)
    # draw box for each img
    # overlay = drawBoxesOnRGB(resize, hits, showLabel=True)
    # cv2.imwrite('images/box.jpg', overlay)
    names = []
    for key, value in hits['BBox'].iteritems():
        x, y, w, h = value[0], value[1], value[2], value[3]
        crop = resize[y:y + h, x:x + w]
        crop_resize = matcher.maintain_aspect_ratio_resize(crop,
                                             height=410)
        final_test = crop_resize[5:5 + 265, 5:5 + 190]
        # download image for test purpose
        # cv2.imwrite('test/' + str(key) + '.jpg', final_test)

        matched = matcher.match_hash(final_test, client.db, store)
        if len(matched) != 0:
            if '-v2' in matched[0]:
                names.append(matched[0].replace('-v2', ''))
            elif '-v3' in matched[0]:
                names.append(matched[0].replace('-v3', ''))
            elif '-v4' in matched[0]:
                names.append(matched[0].replace('-v4', ''))
            elif '-v5' in matched[0]:
                names.append(matched[0].replace('-v5', ''))
            else:
                names.append(matched[0])
        # print("Keys {} with minimum values are : {}".format(key, str(matched)))
    names = list(dict.fromkeys(names))
    response_pickled = jsonpickle.encode({'names': names})
    print("--- % seconds ---" % (time.time() - start))
    try:
        return Response(response=response_pickled, status=200, mimetype="application/json")
    except Exception as error:
        return json.dumps({'error': error})


@app.errorhandler(404)
def page_not_found(e):
    return "<h1>404</h1><p>The resource could not be found.</p>", 404


if __name__ == "__main__":
    try:
        if os.environ.get("MONGO"):
            MONGO_URL = os.environ.get("MONGO")
        app.config["MONGO_URI"] = "mongodb://{}".format(MONGO_URL)
        client = PyMongo(app)
        data = load_hash()
        bulk = store.insert_bulk(client.db, data)
        store.create_index(client.db, "_id")
    except (ServerSelectionTimeoutError, ConnectionFailure, AutoReconnect):
        print('Server not available')
    except Exception as e:
        print(e)
        raise

    app.run(host='0.0.0.0', port=5005, debug=True)
