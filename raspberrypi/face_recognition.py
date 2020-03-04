# -*- coding:utf-8 -*-
import os
import sys
import cv2
import time
import copy
import uuid
import gridfs
import pymongo
import argparse
import datetime
from dotenv import load_dotenv
load_dotenv()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--face_cascade', type=str, default='haarcascade_frontface.xml')
    parser.add_argument('--sleep_second', type=float, default=5)  
    return parser.parse_args()


def get_cpu_serial():
    cpu_serial = '0000000000000000'
    try:
        f = open('/proc/cpuinfo', 'r')
        for line in f:
            if line[0:6] == 'Serial':
                cpu_serial = line[10:26]
        f.close()
    except:
        cpu_serial = 'ERROR000000000'
    return cpu_serial


def db_connection():
    mongo_url = os.getenv('MONGO_URL')
    client = pymongo.MongoClient(mongo_url)
    collection = client.image_collection
    return collection


def main(arguments):
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(arguments.face_cascade)

   # Load CPU Serial
    cpu_serial = get_cpu_serial()

    # Conecction DB
    image_collection = db_connection()
    images = image_collection.images
    fs = gridfs.GridFS(image_collection)

    # video capture
    try:
        cap = cv2.VideoCapture(0)
    except:
        print('Failed to camera loading...')
        return -1
    
    count = 0
    while True:
        try:
            # Read the input image
            ret, img = cap.read()

            # Convert into grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
            # Detection Faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            if len(faces) == 0:
                print('Face not detection...')
                time.sleep(arguments.sleep_second)
                continue
            else:
                print('Detection!')
                print('  ==> count : {}'.format(len(faces)))
            
            temp = []
            for i, (x, y, w, h) in enumerate(faces):
                # crop_img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                crop_img = img[y:y+h, x:x+w]
                suffix = str(uuid.uuid4())
                filepath = '_'.join(['./data/img', suffix]) + '.jpg'

                cv2.imwrite(filepath, crop_img)
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                crop_img = crop_img.tostring()
                img_id = fs.put(crop_img, encoding='utf8')

                temp.append({
                    'cpu_serial': cpu_serial,
                    'image': crop_img,
                    'path': filepath
                })

            # Injection DB
            images.insert_many(temp)
            print('  => Inection DB', end='\n\n')
            
            # Sleep
            time.sleep(arguments.sleep_second)
        except KeyboardInterrupt:
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = get_args()
    main(args)

