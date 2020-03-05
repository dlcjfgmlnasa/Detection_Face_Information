# -*- coding:utf-8 -*-
import os
import time
import pymongo
import requests
from dotenv import load_dotenv
load_dotenv()


def get_collection():
    mongo_url = os.getenv('MONGO_URL')
    client = pymongo.MongoClient(mongo_url)
    collection = client.image_collection
    return collection


def main():
    collection = get_collection()
    images = collection.images

    while True:
        try:
            _cpu_serial_id = None
            ids, file_paths, file_list = [], [], []

            for meta in images.find():
                _id = meta['_id']
                _cpu_serial_id = meta['cpu_serial']
                _image_path = meta['path']

                ids.append(_id)
                file_paths.append(_image_path)
                file_list.append(('image', open(_image_path, 'rb')))
                
            if len(file_list) == 0:
                continue

            try:
                res = requests.post(
                        os.getenv('TARGET_URL'),
                        data={'cpu_serial_id': _cpu_serial_id},
                        files=file_list)
                res.raise_for_status()
            except requests.ConnectionError:
                continue
            except requests.exceptions.HTTPError:
                continue
            except requests.exceptions.RequestException:
                continue

            # delete
            images.delete_many({'_id': {'$in': ids}})
            for path in file_paths:
                os.remove(path)

        except KeyboardInterrupt:
            break


if __name__ == '__main__':
    main()
