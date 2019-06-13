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
from werkzeug.utils import secure_filename
import os
import sys
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
from tensorflow.python.platform import gfile
import argparse
import asyncio
import cv2

has_capacity = False
in_img = None
out_result = None
recog_task = None

def run_recognizer():
    recog_task = await init_recognizer()

def init_recognizer():

def recognize(sess,pnet, rnet, onet,feature_array):
    while True:
        if in_img == None:
            asyncio.sleep(0.03)
        else:
            img = in_img
            in_img = None
            retrieve.recognize_face(sess, pnet, rnet, onet, feature_array)
            

async def cam_routine():
    
    images_placeholder = sess.graph.get_tensor_by_name("input:0")
    images_placeholder = tf.image.resize_images(images_placeholder,(160,160))
    embeddings = sess.graph.get_tensor_by_name("embeddings:0")
    phase_train_placeholder = sess.graph.get_tensor_by_name("phase_train:0")

    image_size = (160, 160)
    embedding_size = embeddings.get_shape()[1]

    cap = cv2.VideoCapture(0)
    task = None
    result = None
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, 0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
        if(gray.size > 0):
            if task != None:
                if task.done():
                    result = task.result()
                    task = None
                    # draw result
            cv2.imshow('img', gray)
            if task == None:
                # task is a future object
                task = loop.run_in_executor(None, retrieve.recognize_async, images_placeholder, phase_train_placeholder, embeddings, sess, pnet, rnet, onet, feature_array, gray)
        asyncio.sleep(0.0333)

def main(args):
    #cam_task = asyncio.create_task(cam_routine())
    loop = asyncio.get_event_loop()
    
    with open(args.pickle,'rb') as f:
        feature_array = pickle.load(f, encoding='utf-8') 
        model_exp = args.model
        graph_fr = tf.Graph()
        sess_fr = tf.Session(graph=graph_fr)
        with graph_fr.as_default():
            with sess_fr.as_default():
                load_model(model_exp)
                pnet, rnet, onet = detect_face.create_mtcnn(sess_fr, None)
                #retrieve.recognize_face(sess_fr,pnet, rnet, onet,feature_array)
                #cam_task = asyncio.create_task(cam_routine())
                loop.run_until_complete(cam_routine())


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle', type=str,
        help='Path for pickle formatted dict to use', default='../extracted_dict.pickle')
    parser.add_argument('--model', type=str,default='../20180408-102900', 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

