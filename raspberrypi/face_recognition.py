# -*- coding:utf-8 -*-
import cv2


if __name__ == '__main__':
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    for line in open('./AFAD-Full/AFAD-Full.txt', encoding='utf-8'):
        print(line)