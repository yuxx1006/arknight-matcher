import pymongo
from pymongo import errors
from pymongo.errors import BulkWriteError
from imagehash import hex_to_hash


class ImageHashStore:

    def __init__(self):
        pass

    # insert image hash to mongodb
    def insert(self, db, hash):
        if db is None:
            raise errors.ConnectionFailure
        try:
            rc = db.imagehashes.insert_one(hash)
        except errors.DuplicateKeyError as e:
            rc = "DuplicateKeyError: '{}'".format(e)
        return rc

    # find image hash from mongodb
    def find(self, db, image_id):
        if db is None:
            raise errors.ConnectionFailure
        rc = db.imagehashes.find_one({
            "_id": image_id
        })
        if not rc:
            return "KeyNotFound"
        return rc

    """
           * store multiple image hash data into mongodb
           * @param data (Array)
           *
           * @return
           """
    def insert_bulk(self, db, data):
        if db is None:
            raise errors.ConnectionFailure
        try:
            db.imagehashes.delete_many({})
            rc = db.imagehashes.insert_many(data)
        except errors.DuplicateKeyError as e:
            rc = "DuplicateKeyError: '{}'".format(e)
        except BulkWriteError as bwe:
            rc = "BulkWriteError: '{}'".format(bwe)

        return rc

    def list_hash(self, db, hashvalue):
        if db is None:
            raise errors.ConnectionFailure
        hash_dic = {}
        try:
            for item in db.imagehashes.find().sort("_id", pymongo.ASCENDING):
                hash_dic[item['operator']] = (hex_to_hash(str(item['_id'])) - hashvalue)
        except errors.PyMongoError:
            return "KeyNotFound"
        return hash_dic

    def create_index(self, db, key):
        db.imagehashes.create_index(key)
