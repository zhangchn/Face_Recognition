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
import socket
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
        'ts': None, # timestamp for img
        'result': [], 
        'result_ts': None, # timestamp for result
        'downsample': 1, 
        'recog_ready': False,
        'ofbbox': [],
        'candidate_area' : [],
        'candidate_ts' : None,
        'of': None
        }

#lock = threading.Lock()

def get_img():
    #with lock:
        return shared['img']

def update_img(img, ts):
    #with lock:
        shared['img'] = img
        shared['ts'] = ts

def get_result():
    #with lock:
        return shared['result']

def update_result(result, t):
    #with lock:
        shared['result'] = result
        shared['result_ts'] = t

def cam_routine():
    
    dt = datetime.datetime
    t = dt.utcnow()
    #cam = 0
    cam = 'rtsp://192.168.10.100/live1.sdp'
    cap = cv2.VideoCapture(cam)
    result = None
    downsample = shared['downsample']
    if sys.platform == 'darwin':
        font = ImageFont.truetype('/System/Library/Fonts/PingFang.ttc', 28)
    else:
        font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 28)
    t = dt.utcnow()
    next_interval = 1
    color1 = (255,255,255)
    color2 = (80, 80, 80)
    failcount = 0
    frametime = []
    # prepare for optical flow
    prev = None
    curr = None
    hsv = None
    of_downsample = 8

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    while True:
        t = dt.utcnow()
        t0 = t
        frametime.append(t)
        if len(frametime) > 5:
            frametime = frametime[1:]
        framerate = "{:.1f} fps".format(
                    (len(frametime) - 1) / (frametime[-1] - frametime[0]).total_seconds()
                ) if len(frametime) > 1 else 'N/A'
        ret, frame = cap.read()
        if frame is None:
            cv2.waitKey(1)
            failcount += 1
            if failcount > 50000:
                cap.release()
                sleep(5)
                cap = cv2.VideoCapture(cam)
                failcount = 0
            continue
        else:
            failcount = 0
            img_copy = frame.copy()

            # confine area to top part
            shape = img_copy.shape
            w = img_copy.shape[1]
            h = img_copy.shape[0] // 3 * 2

            update_img(img_copy[0:h, 0:w, :], dt.utcnow())
            '''
            update_img(img_copy, dt.utcnow())
            '''
        #gray = cv2.cvtColor(frame, 0)
        # pre-downsample
        #gray = cv2.resize(frame, None, fx=0.75, fy=0.75)
        #gray = frame
        #if cv2.waitKey(max(1, int(next_interval * 1000))) & 0xFF == ord('q'):

        '''
        # calculate optical flow in main thread
        img_size = np.asarray(shape)[0:2]
        if curr is None:
            # resize for current frame
            curr = cv2.resize(img_copy, None, fx= 1/of_downsample, fy=1/of_downsample)
            # initialize hsv
            hsv = np.zeros_like(curr)
            hsv[..., 1] = 255
            curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        else:
            prev = curr
            curr = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
            curr = cv2.resize(curr, None, fx= 1/of_downsample, fy=1/of_downsample)
            ofBBox, gray = calc_opt_flow(img_size, curr, prev, hsv, kernel, of_downsample)
            cv2.imshow('gray', gray)

            #previous_faces = get_result()
            #rois.extend([r['box'].copy() for r in previous_faces])
            rois = []
            #of_list = ofBBox[0]
            for bbox in ofBBox:
                temp = []
                if len(rois) == 0:
                    temp.append(bbox)
                else:
                    for j, roi in enumerate(rois):
                        deltaX = np.maximum(roi[0], bbox[0]) - np.minimum(roi[2], bbox[2])
                        deltaY = np.maximum(roi[1], bbox[1]) - np.minimum(roi[3], bbox[3])
                        if deltaX < 0 and deltaY < 0:
                            bbox[0] = np.minimum(roi[0], bbox[0])
                            bbox[1] = np.minimum(roi[1], bbox[1])
                            bbox[2] = np.maximum(roi[2], bbox[2])
                            bbox[3] = np.maximum(roi[3], bbox[3])
                            rois[j][0] = bbox[0]
                            rois[j][1] = bbox[1]
                            rois[j][2] = bbox[2]
                            rois[j][3] = bbox[3]
                        else:
                            temp.append(bbox)
                rois.extend(temp)
            candidate_area = [(bbox, img_copy[bbox[1]:bbox[3], bbox[0]:bbox[2], :]) for bbox in rois]
            shared['candidate_area'] = candidate_area
            shared['candidate_ts'] = t
        '''

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
        print("fps: " + framerate)
        '''
        if(frame.size > 0):
            result = get_result()
            roi_frames = shared['candidate_area']
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
                    cv2.rectangle(frame,(bb[0],bb[1]),(bb[2],bb[3]),color,2)
                    W = int(bb[2]-bb[0])
                    H = int(bb[3]-bb[1])
                    frame_img = Image.fromarray(frame)
                    draw = ImageDraw.Draw(frame_img)
                    draw.text((bb[0] + (W//3), bb[1] - 38), name + ': ' + ("{:.2f}".format(accuracy)), font=font, fill=fill)
                    frame = np.array(frame_img)
                # draw optical flow rects
            for i, rois in enumerate(roi_frames):
                bbox = rois[0]
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,255, 40), 2)
            if not shared['recog_ready']:
                frame_img = Image.fromarray(frame)
                draw = ImageDraw.Draw(frame_img)
                draw.text((frame.shape[1] // 2, frame.shape[0] // 2), "Initializing...", font=font, fill='white')
                frame = np.array(frame_img)

            frame = cv2.resize(frame, None, fx=0.75, fy=0.75)
            cv2.putText(frame, framerate, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255) )
            cv2.imshow('img', frame)
            farneback = shared['of']
            if farneback is not None:
                cv2.imshow('optical flow', farneback)
                '''
        #t1 = dt.utcnow()
        #next_interval = max(0.04 - (t1 - t0).total_seconds(), 0)
        #time.sleep(next_interval)

def recognize(argv):
    dt = datetime.datetime
    
    #detector = dlib.get_frontal_face_detector()
    #detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    use_haar = False#True
    use_dlib = False
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

                if not use_dlib and not use_haar:
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
                    t = dt.utcnow()
                    img = get_img()
                    if img is not None:

                        downsampleShape = (int(img.shape[1] / downsample), int(img.shape[0] / downsample))
                        #img = np.asarray(Image.fromarray(img).resize(downsampleShape, resample=Image.BILINEAR))
                        if downsample > 1:
                            img = cv2.resize(img, downsampleShape)
                        if use_dlib:
                            result = retrieve.recognize_hog(images_placeholder, phase_train_placeholder, embeddings, sess_fr, feature_array, img, detector)
                        elif use_haar:
                            result = retrieve.recognize_haar(images_placeholder, phase_train_placeholder, embeddings, sess_fr, feature_array, img, detector)
                        else:
                            result = []
                            for roi in shared['candidate_area']:
                                bbox = roi[0]
                                cropped = roi[1]
                                #print("align in {},{}: {},{}".format(bbox[0], bbox[1], bbox[2], bbox[3]))
                                r = retrieve.recognize_mtcnn(images_placeholder, phase_train_placeholder, embeddings, sess_fr, pnet, rnet, onet, feature_array, cropped)
                                for i, face in enumerate(r):
                                    b = face['box']
                                    b[0] += bbox[0]
                                    b[1] += bbox[1]
                                    b[2] += bbox[0]
                                    b[3] += bbox[1]
                                    r[i]['box'] = b
                                    print("face: {},{},{},{}".format(b[0], b[1], b[2], b[3]))
                                result.extend(r)
                        update_result(result, t)

def calc_opt_flow(img_size, curr, prev, hsv, kernel, downsample):
    flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 2, 7, 1.5, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    of_threshold = 30
    gray = cv2.threshold(gray, of_threshold, 255, cv2.THRESH_BINARY)[1]
    #shared['of'] = gray
    contours, hier = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ofBBox = []
    margin = 10 // downsample
    for c in contours:
        if cv2.contourArea(c) < 800:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        bb = np.zeros(4, dtype=np.int32)
        #expand with a fixed margin
        bb[0] = np.maximum(x - margin, 0) * downsample
        bb[1] = np.maximum(y - margin, 0) * downsample
        bb[2] = np.minimum((x+w+margin) * downsample, img_size[1])
        bb[3] = np.minimum((y+h+margin) * downsample, img_size[0])
        ofBBox.append(bb)
    return ofBBox, gray

def opt_flow():
    prev = None
    curr = None

    hsv = None

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    downsample = 8
    dt = datetime.datetime
    prev_ts = dt.utcfromtimestamp(0)
    while True:
        img = shared['img']
        ts = shared['ts']
        if img is None or ts == prev_ts:
            sleep(0.01)
            continue
        img_size = np.asarray(img.shape)[0:2]
        if curr is None:
            # resize for current frame
            curr = cv2.resize(img, None, fx= 1/downsample, fy=1/downsample)
            # initialize hsv
            hsv = np.zeros_like(curr)
            hsv[..., 1] = 255
            curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
            continue
        else:
            prev = curr
            curr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            curr = cv2.resize(curr, None, fx= 1/downsample, fy=1/downsample)

        ofBBox, gray = calc_opt_flow(img_size, curr, prev, hsv, kernel, downsample)
        if len(ofBBox) > 0:
            shared['ofbbox'].insert(0, (ofBBox, img))
        if len(shared['ofbbox']) > 3:
            shared['ofbbox'].pop()
        of_frames = shared['ofbbox']
        rois = []
        if len(of_frames) > 0:
            previous_faces = get_result()
            rois.extend([r['box'].copy() for r in previous_faces])
            of_list = of_frames[0][0]
            for bbox in of_list:
                temp = []
                if len(rois) == 0:
                    temp.append(bbox)
                else:
                    for j, roi in enumerate(rois):
                        deltaX = np.maximum(roi[0], bbox[0]) - np.minimum(roi[2], bbox[2])
                        deltaY = np.maximum(roi[1], bbox[1]) - np.minimum(roi[3], bbox[3])
                        if deltaX < 0 and deltaY < 0:
                            bbox[0] = np.minimum(roi[0], bbox[0])
                            bbox[1] = np.minimum(roi[1], bbox[1])
                            bbox[2] = np.maximum(roi[2], bbox[2])
                            bbox[3] = np.maximum(roi[3], bbox[3])
                            rois[j][0] = bbox[0]
                            rois[j][1] = bbox[1]
                            rois[j][2] = bbox[2]
                            rois[j][3] = bbox[3]
                        else:
                            temp.append(bbox)
                rois.extend(temp)
            candidate_area = [(bbox, img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]) for bbox in rois]
            #print("roi count: {}".format(len(rois)))
            shared['candidate_area'] = candidate_area
            shared['candidate_ts'] = ts
        sleep(0.04)

def sample_saver():
    dt = datetime.datetime
    prev_ts = dt.utcfromtimestamp(0)
    while True:
        rs = shared['result']
        ts = shared['result_ts']
        if rs is not None and ts != prev_ts:
            for i, r in enumerate(rs):
                img = r['faceimg']
                name = r['name']
                acc = r['acc']
                cv2.imwrite('/tmp/{}{:02}{:02}_{:02}{:02}{:02}_{:06}_{}_{:.04}_{}.png'.format(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second, ts.microsecond, name, acc, i), img)
        prev_ts = ts
        sleep(0.033)

def roi_sender():
    dt = datetime.datetime
    prev_ts = dt.utcfromtimestamp(0)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect('/tmp/img_sock')
    except socket.error as msg:
        print("socket error: {}".format(msg))
    while True:
        ts = shared['candidate_ts']
        if ts == prev_ts:
            #print("sleep")
            sleep(0.03)
            continue
        prev_ts = ts
        candidate_area = shared['candidate_area']
        for i, (bbox, img) in enumerate(candidate_area):
            #t0 = dt.utcnow()
            img_bytes = pickle.dumps(img)
            try:
                #t1 = dt.utcnow()
                sock.send("{}:{}\n".format(i, len(img_bytes)).encode())
                #t2 = dt.utcnow()
                sock.sendall(img_bytes)
                #t3 = dt.utcnow()
                #print("t1: {:.3f}, t2: {:.3f}, t3: {:.3f}".format(
                #    (t1 - t0).total_seconds(),
                #    (t2 - t1).total_seconds(),
                #    (t3 - t2).total_seconds()))
            except:
                sock.close()
                try:
                    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    sock.connect('/tmp/img_sock')
                except socket.error as msg:
                    print("socket error: {}".format(msg))
            '''
            path = '/tmp/cand_{}{:02}{:02}_{:02}{:02}{:02}_{:06}_{}.png'.format(
                ts.year, ts.month, 
                ts.day, ts.hour, ts.minute, 
                ts.second, ts.microsecond, i)
            cv2.imwrite(path, img)
            '''

def main(args):
    thread = threading.Thread(target=recognize, args=[args])
    thread2 = threading.Thread(target=opt_flow)
    thread3 = threading.Thread(target=roi_sender)
    #thread3 = threading.Thread(target=sample_saver)

    #thread.start()
    thread2.start()
    thread3.start()
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

