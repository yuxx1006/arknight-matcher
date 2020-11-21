try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    import Image
import cv2
from flask import Flask, request, Response, jsonify
from flask_pymongo import PyMongo, MongoClient
from pymongo.errors import ConnectionFailure, AutoReconnect, ServerSelectionTimeoutError
import urllib.request
from urllib.error import URLError, HTTPError
import imagehash
import json
import os
import numpy as np
import jsonpickle
import time
from datetime import timedelta
from modules.match import Match
from stores.image_hash_store import ImageHashStore
from MTM import matchTemplates, drawBoxesOnRGB

MONGO_URL = "127.0.0.1:27017/arknights"
client = None
store = ImageHashStore()
IMAGE_LOCAL = './images/local-filename.'
TEMPLATE_PATH = './template/'
DEFAULT_IMAGE_HEIGHT = 1080
NAME_DICT = {}
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
    with open('home.html', encoding='utf-8', errors='ignore') as f:
        r = f.read()
    return  r


@app.route('/find-operators', methods=['POST'])
def get_image_classification():
    start = time.time()
    try:
        r = request
        nparr = np.frombuffer(r.files['image'].read(), np.uint8)
        # nparr = np.frombuffer(r.data, np.uint8)
        test_img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        test_img = test_img[:,:,:3]
    except Exception as e:
        raise RuntimeError("Failed to load image data")

    matcher = Match()
    # resize img to match template
    default = matcher.maintain_aspect_ratio_resize(test_img, height=DEFAULT_IMAGE_HEIGHT)
    marker_ratio = matcher.find_ratio(default)
    resize = matcher.maintain_aspect_ratio_resize(default,
                                                  height=int(DEFAULT_IMAGE_HEIGHT / float(marker_ratio)))

    # image binary for box detection
    black_img = resize.copy()
    black_img_h, black_img_w = black_img.shape[:2]
    mask = np.zeros((black_img_h + 2, black_img_w + 2), dtype=np.uint8)
    for p in [(black_img_w - 1, 0), (black_img_w - 1, black_img_h - 1)]:
        cv2.floodFill(black_img, mask, p, [255, 255, 255])
    THRES = 60
    black_img[black_img <= THRES] = 0
    black_img[black_img > THRES] = 255
    black_img = cv2.cvtColor(black_img, cv2.COLOR_BGR2GRAY)
    black_img[black_img != 0] = 255
    cv2.imwrite("static/black.jpg", black_img)

    hits = matcher.match_template(black_img)
    # draw box for each img
    # overlay = drawBoxesOnRGB(resize, hits, showLabel=True)
    # cv2.imwrite('images/box.jpg', overlay)
    names = []
    result_img = resize.copy()
    for key, value in hits['BBox'].iteritems():
        x, y, w, h = value[0], value[1] - 264, value[2], value[3] + 264
        crop = resize[y:y + h, x:x + w]
        if len(crop) == 0: continue
        crop_resize = matcher.maintain_aspect_ratio_resize(crop,
                                             height=410)
        final_test = crop_resize[5:5 + 265, 5:5 + 190]
        cv2.imwrite("./static/%s.jpg" % key, final_test) # if recognition is wrong, add sample to "template/"

        # download image for test purpose
        # cv2.imwrite('test/' + str(key) + '.jpg', final_test)

        result_img = cv2.rectangle(result_img, (x, y), (x + w, y + h), [128, 0, 0], thickness=3)
        matched = matcher.match_hash(final_test, client.db, store)
        if len(matched) != 0:
            if matched[0] == '00': continue
            if '-v' in matched[0]:
                p = matched[0].find('-v')
                id = matched[0][:p]
            else:
                id = matched[0]
            name = NAME_DICT[id]
            names.append(name)

            result_img = cv2.rectangle(result_img, (x, y), (x + w, y + 70), [128, 0, 0], thickness=-1)
            result_img = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(result_img)
            fontStyle = ImageFont.truetype("font/simsun.ttc", 50, encoding="utf-8")
            draw.text((x, y), name, (255, 255, 255), font=fontStyle)
            result_img = cv2.cvtColor(np.asarray(result_img), cv2.COLOR_RGB2BGR)

    result_img = matcher.maintain_aspect_ratio_resize(result_img, width=1024)
    cv2.imwrite("./static/result.jpg", result_img)
        # print("Keys {} with minimum values are : {}".format(key, str(matched)))
    names = list(dict.fromkeys(names))
    response_pickled = jsonpickle.encode({'names': names})
    print("--- %s seconds ---" % (time.time() - start))
    print(names)

    with open('result.html', encoding='utf-8', errors='ignore') as f:
        response = f.read()
    return  response


@app.errorhandler(404)
def page_not_found(e):
    return "<h1>404</h1><p>The resource could not be found.</p>", 404

def load_dict(file_path):
    with open(file_path, encoding='utf-8', errors='ignore') as file:
        for line in file.readlines():
            p = line.find(',')
            NAME_DICT[line[:p]] = line[p+1:-1]

if __name__ == "__main__":
    os.chdir('D:/GitHub/arknight-matcher/')
    try:
        if os.environ.get("MONGO"):
            MONGO_URL = os.environ.get("MONGO")
        app.config["MONGO_URI"] = "mongodb://{}".format(MONGO_URL)
        client = PyMongo(app)
        data = load_hash()
        load_dict("arknights.csv")
        bulk = store.insert_bulk(client.db, data)
        store.create_index(client.db, "_id")
    except (ServerSelectionTimeoutError, ConnectionFailure, AutoReconnect):
        print('Server not available')
    except Exception as e:
        print(e)
        raise

    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)
    app.run(host='0.0.0.0', port=5005, debug=True)
