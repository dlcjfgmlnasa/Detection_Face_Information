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
            file_list = []
            for meta in images.find():
                _id = meta['_id']
                _cpu_serial_id = meta['cpu_serial']
                _image_path = meta['path']

                file_list.append(('image', open(_image_path, 'rb')))
                
                # remove information (mongo, database)
                images.delete_one({'_id': _id})
                os.remove(_image_path)

            if len(file_list) == 0:
                continue

            url = os.getenv('TARGET_URL') + 'datacollections/image/'
            res = requests.post(
                url,
                data={'cpu_serial_id': _cpu_serial_id},
                files=file_list
            )
        except KeyboardInterrupt:
            break


if __name__ == '__main__':
    main()
