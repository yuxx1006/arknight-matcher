from gevent import monkey
monkey.patch_all()
from gevent.pywsgi import WSGIServer
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    import Image
import cv2
from flask import Flask, request, Response
from flask_pymongo import PyMongo
from pymongo.errors import ConnectionFailure, AutoReconnect, ServerSelectionTimeoutError
import imagehash
import json
import os
import numpy as np
import time
from datetime import timedelta
from modules.match import Match
from stores.image_hash_store import ImageHashStore
from MTM import matchTemplates, drawBoxesOnRGB
import random

MONGO_URL = "127.0.0.1:27017/arknights"
client = None
store = ImageHashStore()
IMAGE_LOCAL = './images/local-filename.'
TEMPLATE_PATH = './template/'
DEFAULT_IMAGE_HEIGHT = 1080
NAME_DICT = {}
DEBUG_MODE = False
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


@app.route('/images/<image_name>', methods=['POST'])
def insert_hash(image_name):
    global client
    try:
        r = request
        nparr = np.frombuffer(r.files['image'].read(), np.uint8)
        test_img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        test_img = test_img[:,:,:3]
    except Exception as e:
        raise RuntimeError("Failed to load image data")
    pil_image = Image.fromarray(test_img)
    hashvalue = imagehash.dhash(pil_image)
    hashes = {"_id": str(hashvalue), "operator": image_name}
    # save the image hash
    try:
        if client is None:
            app.config["MONGO_URI"] = "mongodb://{}".format(os.environ.get("MONGO"))
            client = PyMongo(app)
        rs = store.insert(client.db, hashes)
        print(rs)
        store.create_index(client.db, "_id")
        json_response = json.dumps(hashes, indent=4, sort_keys=True, ensure_ascii=False)
    except (AutoReconnect, ServerSelectionTimeoutError) as error:
        raise RuntimeError('server not available: {}'.format(error))
    except Exception as e:
        raise e
    response = Response(json_response, content_type='application/json; charset=utf-8')
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.status_code = 200
    return response


@app.route('/matcher', methods=['POST'])
def get_image_classification():
    start = time.time()
    try:
        r = request
        nparr = np.frombuffer(r.files['image'].read(), np.uint8)
        test_img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        test_img = test_img[:,:,:3]
    except Exception as e:
        raise RuntimeError("Failed to load image data")

    random_id = random.randint(0,9999)
    cv2.imwrite("./static/origin%s.jpg"%random_id, test_img)
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
    if DEBUG_MODE: cv2.imwrite("./static/black%s.jpg"%random_id, black_img)

    hits = matcher.match_template(black_img)
    names = []
    result_img = resize.copy()

    for key, value in hits['BBox'].iteritems():
        x, y, w, h = value[0], value[1]-262, 205, 410
        crop = resize[y:y + h, x:x + w]
        if len(crop) == 0: continue
        crop_resize = matcher.maintain_aspect_ratio_resize(crop,
                                             height=410)
        final_test = crop_resize[5:5 + 265, 5:5 + 190]
        if DEBUG_MODE: cv2.imwrite("./static/test/%s.jpg" % key, final_test) # if recognition is wrong, add sample to "template/"

        # result_img = cv2.rectangle(result_img, (x, y), (x + w, y + h), [128, 0, 0], thickness=3)
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

            result_img = cv2.rectangle(result_img, (x, y), (x + w, y + h), [128, 0, 0], thickness=3)
            result_img = cv2.rectangle(result_img, (x, y), (x + w, y + 70), [128, 0, 0], thickness=-1)
            result_img = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(result_img)
            fontStyle = ImageFont.truetype("font/simsun.ttc", 50, encoding="utf-8")
            draw.text((x, y), name, (255, 255, 255), font=fontStyle)
            result_img = cv2.cvtColor(np.asarray(result_img), cv2.COLOR_RGB2BGR)

    result_img = matcher.maintain_aspect_ratio_resize(result_img, width=1024)
    cv2.imwrite("./static/result%s.jpg"%random_id, result_img)
    names = list(dict.fromkeys(names))
    print("--- %s seconds ---" % (time.time() - start))
    print(names)

    with open('result.html', encoding='utf-8', errors='ignore') as f:
        response = f.read()
        response = response.replace("static/result.jpg","static/result%s.jpg"%random_id)
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
    # os.chdir('D:/GitHub/arknight-matcher/')
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
    # http_server = WSGIServer(('0.0.0.0', 5005), app)
    # http_server.serve_forever()