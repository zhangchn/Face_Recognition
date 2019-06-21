#!flask/bin/python
################################################################################################################################
#------------------------------------------------------------------------------------------------------------------------------                                                                                                                             
# This file implements the REST layer. It uses flask micro framework for server implementation. Calls from front end reaches 
# here as json and being branched out to each projects. Basic level of validation is also being done in this file. #                                                                                                                                  	       
#-------------------------------------------------------------------------------------------------------------------------------                                                                                                                              
################################################################################################################################
#from flask import Flask, jsonify, abort, request, make_response, url_for,redirect, render_template
#from flask.ext.httpauth import HTTPBasicAuth
#from flask_httpauth import HTTPBasicAuth
import os
import sys
import ctypes
import mmap
import struct
import random
from tensorflow.python.platform import gfile
from six import iteritems
sys.path.append('..')
import numpy as np
from lib.src import retrieve
from lib.src.align import detect_face
from lib.src.facenet import load_model
import tensorflow as tf
import pickle
import json
from tensorflow.python.platform import gfile
import argparse
import threading
import dlib
import cv2
from PIL import ImageFont, ImageDraw, Image
import datetime
from time import sleep

in_img = None
out_result = None

shared = {
        'img': None, 
        'result': None, 
        'downsample': 1, 
        'recog_ready': False,
        'ofbbox': [],
        }

#lock = threading.Lock()

def get_img():
    #with lock:
        return shared['img']

def update_img(img):
    #with lock:
        shared['img'] = img

def get_result():
    #with lock:
        return shared['result']

def update_result(result):
    #with lock:
        shared['result'] = result

def cam_routine():
    
    dt = datetime.datetime
    t = dt.utcnow()
    cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture('rtsp://192.168.10.100/live1.sdp')
    result = None
    downsample = shared['downsample']
    if sys.platform == 'darwin':
        font = ImageFont.truetype('/System/Library/Fonts/PingFang.ttc', 28)
    else:
        font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.tcc', 28)
    t = dt.utcnow()
    next_interval = 1
    color1 = (255,255,255)
    color2 = (80, 80, 80)
    while True:
        t = dt.utcnow()
        t0 = t
        ret, frame = cap.read()
        if frame is None:
            cv2.waitKey(1)
            continue
        #gray = cv2.cvtColor(frame, 0)
        # pre-downsample
        #gray = cv2.resize(frame, None, fx=0.75, fy=0.75)
        gray = frame
        #if cv2.waitKey(max(1, int(next_interval * 1000))) & 0xFF == ord('q'):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
        if(gray.size > 0):
            result = get_result()
            if result is not None:
                # draw face rects and result
                for i, r in enumerate(result):
                    bb = r['box'].copy()
                    name = r['name']
                    accuracy = r['acc']
                    if accuracy >= 0.9:
                        name = name + '??'
                        color = color2
                        fill = 'gray'
                    else:
                        color = color1
                        fill = 'white'
                    for i, v in enumerate(bb):
                        bb[i] = int(v * downsample)
                    cv2.rectangle(gray,(bb[0],bb[1]),(bb[2],bb[3]),color,2)
                    W = int(bb[2]-bb[0])
                    H = int(bb[3]-bb[1])
                    gray_img = Image.fromarray(gray)
                    draw = ImageDraw.Draw(gray_img)
                    draw.text((bb[0] + (W//3), bb[1] - 38), name + ': ' + ("{:.2f}".format(accuracy)), font=font, fill=fill)
                    gray = np.array(gray_img)
                # draw optical flow rects
                for i, bbox in enumerate(shared['ofbbox']):
                    cv2.rectangle(gray, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,255, 0), 2)
            elif not shared['recog_ready']:
                gray_img = Image.fromarray(gray)
                draw = ImageDraw.Draw(gray_img)
                draw.text((gray.shape[1] // 2, gray.shape[0] // 2), "Initializing...", font=font, fill='white')
                gray = np.array(gray_img)

            update_img(gray)
            cv2.imshow('img', gray)
        t1 = dt.utcnow()
        next_interval = max(0.04 - (t1 - t0).total_seconds(), 0)
        #time.sleep(next_interval)

def recognize(argv):
    dt = datetime.datetime
    
    detector = dlib.get_frontal_face_detector()
    with open(argv.pickle,'rb') as f:
        sys.stderr.write("will load feature\n")
        feature_array = pickle.load(f, encoding='utf-8') 
        model_exp = argv.model
        graph_fr = tf.Graph()
        sess_fr = tf.Session(graph=graph_fr)
        with graph_fr.as_default():
            with sess_fr.as_default():
                image_size = (160, 160)
                sys.stderr.write("will load model\n")
                #sys.stderr.write("did load model\n")
                pnet, rnet, onet = detect_face.create_mtcnn(sess_fr, None)
                load_model(model_exp)
                #sys.stderr.write("will get placeholders\n")
                images_placeholder = sess_fr.graph.get_tensor_by_name("input:0")
                images_placeholder = tf.image.resize_images(images_placeholder,image_size)
                embeddings = sess_fr.graph.get_tensor_by_name("embeddings:0")
                phase_train_placeholder = sess_fr.graph.get_tensor_by_name("phase_train:0")

                downsample = shared['downsample']
                shared['recog_ready'] = True
                while True:
                    img = get_img()
                    if img is not None:

                        downsampleShape = (int(img.shape[1] / downsample), int(img.shape[0] / downsample))
                        #img = np.asarray(Image.fromarray(img).resize(downsampleShape, resample=Image.BILINEAR))
                        if downsample > 1:
                            img = cv2.resize(img, downsampleShape)
                        result = retrieve.recognize_mtcnn(images_placeholder, phase_train_placeholder, embeddings, sess_fr, pnet, rnet, onet, feature_array, img)
                        #result = retrieve.recognize_hog(images_placeholder, phase_train_placeholder, embeddings, sess_fr, feature_array, img, detector)
                        update_result(result)

def opt_flow():
    prev = None
    curr = None

    hsv = None

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    downsample = 8
    while True:
        if curr is None:
            sleep(0.03)
            if shared['img'] is not None:
                curr = shared['img']
                # resize curr
                curr = cv2.resize(curr, None, fx= 1/downsample, fy=1/downsample)
                # initialize hsv
                hsv = np.zeros_like(curr)
                hsv[..., 1] = 255
                curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
            continue
        if shared['img'] is not None:
            prev = curr
            curr = shared['img']
            curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
            curr = cv2.resize(curr, None, fx= 1/downsample, fy=1/downsample)
        else:
            sleep(0.033)
            continue

        flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 2, 7, 1.5, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        gray = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)[1]
        contours, hier = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ofBBox = []
        for c in contours:
            if cv2.contourArea(c) < 800:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = x * downsample
            bb[1] = y * downsample
            bb[2] = (x+w) * downsample
            bb[3] = (y+h) * downsample
            ofBBox.append(bb)
        shared['ofbbox'] = ofBBox
        sleep(0.033)


def main(args):
    thread = threading.Thread(target=recognize, args=[args])
    thread.start()
    thread2 = threading.Thread(target=opt_flow)
    thread2.start()
    cam_routine()



def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle', type=str,
        help='Path for pickle formatted dict to use', default='../extracted_dict.pickle')
    parser.add_argument('--model', type=str,default='../20180408-102900', 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    return parser.parse_args(argv)

if __name__ == '__main__':
    #loop = asyncio.get_event_loop()
    main(parse_arguments(sys.argv[1:]))

