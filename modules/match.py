try:
    from PIL import Image
except ImportError:
    import Image
import imagehash
import cv2
import numpy as np
from MTM import matchTemplates

# constant
TEMPLATE_IMG = './images/marker.jpg'
TEMPLATE_IMG_1 = './images/t1.jpg'


class Match:
    def __init__(self):
        pass

    """
            * resize a image and maintains aspect ratio
            * @param filename (String)
            *
            * @return
            *   $image (ARRAY)
            """
    def maintain_aspect_ratio_resize(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        # Grab the image size and initialize dimensions
        dim = None
        (h, w) = image.shape[:2]

        # Return original image if no need to resize
        if width is None and height is None:
            return image

        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        # Return the resized image
        return cv2.resize(image, dim, interpolation=inter)

    """
                * find resize ratio for given image
                * @param filename (String)
                *
                * @return
                *   $image (ARRAY)
                """
    def find_ratio(self, image):
        # convert given image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mark = cv2.imread(TEMPLATE_IMG)
        mark = cv2.cvtColor(mark, cv2.COLOR_BGR2GRAY)
        (tH, tW) = mark.shape[:2]

        found = None
        for scale in np.linspace(0.5, 2, 20)[::-1]:
            # resize the image according to the scale
            resized = self.maintain_aspect_ratio_resize(gray, height=int(gray.shape[0]*scale))
            r = gray.shape[1] / float(resized.shape[1])
            # if the resized image is smaller than the template, then break
            # from the loop
            if resized.shape[0] < tH or resized.shape[1] < tW:
                break
            result = cv2.matchTemplate(resized, mark, cv2.TM_CCOEFF_NORMED)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
            # if we have found a new maximum correlation value, then update
            # the bookkeeping variable
            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r)

        (_, maxLoc, r) = found
        return r

    """
           * match template for given image
           * @param resized image
           *
           * @return
           *   $names pd series with key, bbox
           """
    def match_template(self, resize):
        t1 = cv2.imread(TEMPLATE_IMG_1, cv2.IMREAD_UNCHANGED)
        list_template = [('t1', t1)]
        hits = matchTemplates(list_template, resize, score_threshold=0.4, method=cv2.TM_CCOEFF_NORMED, maxOverlap=0.1)
        return hits

    """
           * match hash value for given image
           * @param cropped image
           * @param db - community database
           * @param store - hash store
           *
           * @return
           *   $hash value with minimum difference from the given image
           """
    def match_hash(self, crop, db, store):
        pil_image = Image.fromarray(crop)
        hashvalue = imagehash.dhash(pil_image)
        hash_values = store.list_hash(db, hashvalue)
        hash_sort = {k: v for k, v in sorted(hash_values.items(), key=lambda item: item[1])}
        temp = min(hash_sort.values())
        res = [key for key in hash_sort if (hash_sort[key] == temp and temp < 14)]
        return res

